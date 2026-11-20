// mla_test.hip.cpp - unit tests + timings for mla_attn.hip.cpp (Hy4-Flash MLA attention, HIP / CUDA)
// usage: mla_test [--model <Hy4-Flash-SQ-64.strata>] [--perf] [--noreal] [--layers 5,2]
#include "hy4_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "hy4_json.h"
#include "mla_attn.hip.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

using namespace hy4;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() {}
    explicit Dev(size_t n_) { alloc(n_); }
    void alloc(size_t n_) { free_(); n = n_; CK(hipMalloc((void**)&p, n * sizeof(T))); }
    void free_() { if (p) hipFree(p); p = nullptr; n = 0; }
    ~Dev() { free_(); }
    void up(const std::vector<T>& v) { if (n != v.size()) alloc(v.size()); CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); }
    void down(std::vector<T>& v) const { v.resize(n); CK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); }
};

static uint16_t f2h(float f) {
    __half h = __float2half(f);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float h2f(uint16_t u) {
    __half h;
    memcpy(&h, &u, 2);
    return __half2float(h);
}

struct Weights {
    int NH = 32;
    std::vector<float> wk;     // [NH][192][512]  (fp32 truth; GPU gets fp16 rounding)
    std::vector<float> wv;     // [NH][256][512]
    std::vector<float> normw;  // [512]
    std::vector<float> sinks;  // [NH]
    std::vector<uint8_t> kvb_raw;  // raw S_K kv_b_proj bytes (real layers only)
    int kvb_fmt = 0;
    double theta = 1e7;
};

static void make_random_weights(Weights& W, int NH, std::mt19937_64& rng, double theta) {
    W.NH = NH;
    W.theta = theta;
    std::normal_distribution<float> nd(0.f, 1.f);
    W.wk.resize((size_t)NH * 192 * 512);
    W.wv.resize((size_t)NH * 256 * 512);
    for (auto& x : W.wk) x = h2f(f2h(nd(rng) * 0.04f));
    for (auto& x : W.wv) x = h2f(f2h(nd(rng) * 0.04f));
    W.normw.resize(512);
    for (auto& x : W.normw) x = 0.6f + 0.4f * nd(rng);
    W.sinks.resize(NH);
    for (auto& x : W.sinks) x = 0.7f * nd(rng);
}

// ---- real layer weights from the .strata ----
static bool load_real_layer(Weights& W, const std::string& path, int layer, std::string& err) {
    using namespace strata;
    artifact::StrataFileReader rd;
    if (!rd.open(path, err)) return false;
    hy4::JVal man = hy4::json_parse(rd.manifest_json());
    const auto& tc = man.at("config").at("text_config");
    bool swa = tc.at("layer_types")[layer].str() == "sliding_attention";
    int NH = swa ? (int)tc.at("swa_num_attention_heads").i() : (int)tc.at("num_attention_heads").i();
    W.NH = NH;
    W.theta = swa ? tc.at("rope_parameters").at("sliding_attention").at("rope_theta").f() : tc.at("rope_parameters").at("deepseek_sparse_attention").at("rope_theta").f();
    uint64_t sec1 = rd.header().sections[1].file_offset;
    auto find = [&](const std::string& name, std::string& fmt, std::vector<int64_t>& shape, uint64_t& off, uint64_t& bytes) {
        for (auto& t : man.at("trunk_tensors").a)
            if (t.at("name").str() == name) {
                fmt = t.at("format").str();
                shape.clear();
                for (auto& d : t.at("shape").a) shape.push_back(d.i());
                off = (uint64_t)t.at("offset").i();
                bytes = (uint64_t)t.at("bytes").i();
                return true;
            }
        return false;
    };
    std::string pre = "model.layers." + std::to_string(layer) + ".self_attn.";
    std::string fmt;
    std::vector<int64_t> shape;
    uint64_t off, bytes;
    if (!find(pre + "kv_b_proj.weight", fmt, shape, off, bytes)) { err = "no kv_b_proj"; return false; }
    const std::string kvfmt = fmt;
    if (fmt != "S5_K" && fmt != "S4_K" && fmt != "S6_K" && fmt != "S8_K") { err = "unexpected kv_b format " + fmt; return false; }
    strata::quant::sk::SkCode code = fmt == "S5_K" ? strata::quant::sk::SkCode::S5_K : fmt == "S4_K" ? strata::quant::sk::SkCode::S4_K : fmt == "S6_K" ? strata::quant::sk::SkCode::S6_K : strata::quant::sk::SkCode::S8_K;
    std::vector<uint8_t> raw(bytes);
    if (!rd.read_at(sec1 + off, raw.data(), bytes, err)) return false;
    W.kvb_raw = raw;
    W.kvb_fmt = fmt == "S5_K" ? 5 : fmt == "S4_K" ? 4 : fmt == "S6_K" ? 6 : 8;
    int rows = (int)shape[0], cols = (int)shape[1];
    if (rows != NH * 448 || cols != 512) { err = "kv_b shape mismatch"; return false; }
    size_t rb = (size_t)(cols / 256) * strata::quant::sk::superblock_bytes(code);
    W.wk.assign((size_t)NH * 192 * 512, 0.f);
    W.wv.assign((size_t)NH * 256 * 512, 0.f);
    std::vector<float> row(cols);
    for (int r = 0; r < rows; ++r) {
        strata::quant::sk::dequant_sk_row(raw.data() + (size_t)r * rb, code, row.data(), cols);
        int h = r / 448, i = r % 448;
        float* dst = i < 192 ? &W.wk[((size_t)h * 192 + i) * 512] : &W.wv[((size_t)h * 256 + (i - 192)) * 512];
        memcpy(dst, row.data(), cols * 4);
    }
    auto getf = [&](const std::string& name, std::vector<float>& out) {
        if (!find(name, fmt, shape, off, bytes) || fmt != "F32") return false;
        out.resize(bytes / 4);
        return rd.read_at(sec1 + off, out.data(), bytes, err);
    };
    if (!getf(pre + "kv_a_layernorm.weight", W.normw)) { err = "no kv_a_layernorm"; return false; }
    if (!getf(pre + "learnable_sink_param", W.sinks)) { err = "no sinks"; return false; }
    printf("  loaded real layer %d: %s heads=%d theta=%g kv_b %s [%d,%d]  sinks[0..3]=%.3f %.3f %.3f  normw[0..2]=%.3f %.3f %.3f\n", layer, swa ? "SWA" : "sparse", NH, W.theta, kvfmt.c_str(), rows, cols,
           W.sinks[0], W.sinks[1], W.sinks[2], W.normw[0], W.normw[1], W.normw[2]);
    return true;
}

// ---- CPU double reference (non-absorbed, as tools/squant/hy_v4_flash.py) ----
static void rope_ref(double* x, int D, long pos, double theta) {
    for (int i = 0; i < D / 2; ++i) {
        float freq = 1.0f / powf((float)theta, (float)(2 * i) / (float)D);  // float32 like torch
        float ang = (float)pos * freq;
        double c = cosf(ang), s = sinf(ang);
        double x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

struct RefIn {
    int M, P, p0, mode, window;
    const float* q_b;                       // [M][NH][256]
    const float* kv_a;                      // [P][576]
    const float* gate;                      // [M][NH*256] or null
    const std::vector<int32_t>* keys;       // list mode [M][kmax]
    const std::vector<int32_t>* nkeys;
    int kmax;
};

static void ref_attention(const Weights& W, const RefIn& in, std::vector<double>& o_ref) {
    const int NH = W.NH, M = in.M, P = in.P;
    std::vector<double> kvn((size_t)P * 512), kpe((size_t)P * 64);
    for (int j = 0; j < P; ++j) {
        const float* r = in.kv_a + (size_t)j * 576;
        double ss = 0;
        for (int c = 0; c < 512; ++c) ss += (double)r[c] * r[c];
        double inv = 1.0 / std::sqrt(ss / 512.0 + 1e-5);
        for (int c = 0; c < 512; ++c) kvn[(size_t)j * 512 + c] = (double)r[c] * inv * W.normw[c];
        for (int c = 0; c < 64; ++c) kpe[(size_t)j * 64 + c] = r[512 + c];
        rope_ref(&kpe[(size_t)j * 64], 64, j, W.theta);
    }
    // k_nope / v for every position and head (the non-absorbed expansion)
    std::vector<double> knope((size_t)P * NH * 192), vv((size_t)P * NH * 256);
    for (int j = 0; j < P; ++j)
        for (int h = 0; h < NH; ++h) {
            const double* kv = &kvn[(size_t)j * 512];
            for (int i = 0; i < 192; ++i) {
                const float* w = &W.wk[((size_t)h * 192 + i) * 512];
                double s = 0;
                for (int c = 0; c < 512; ++c) s += (double)w[c] * kv[c];
                knope[((size_t)j * NH + h) * 192 + i] = s;
            }
            for (int i = 0; i < 256; ++i) {
                const float* w = &W.wv[((size_t)h * 256 + i) * 512];
                double s = 0;
                for (int c = 0; c < 512; ++c) s += (double)w[c] * kv[c];
                vv[((size_t)j * NH + h) * 256 + i] = s;
            }
        }
    o_ref.assign((size_t)M * NH * 256, 0.0);
    std::vector<double> sc;
    for (int t = 0; t < M; ++t) {
        long pos = in.p0 + t;
        for (int h = 0; h < NH; ++h) {
            const float* qb = in.q_b + ((size_t)t * NH + h) * 256;
            double qn[192], qp[64];
            for (int i = 0; i < 192; ++i) qn[i] = qb[i];
            for (int i = 0; i < 64; ++i) qp[i] = qb[192 + i];
            rope_ref(qp, 64, pos, W.theta);
            std::vector<int> cand;
            if (in.mode == MLA_CAUSAL) for (int j = 0; j <= pos; ++j) cand.push_back(j);
            else if (in.mode == MLA_WINDOW) for (long j = std::max<long>(0, pos - in.window + 1); j <= pos; ++j) cand.push_back((int)j);
            else for (int i = 0; i < (*in.nkeys)[t]; ++i) cand.push_back((*in.keys)[(size_t)t * in.kmax + i]);
            sc.assign(cand.size(), 0.0);
            double mx = W.sinks[h];
            for (size_t k = 0; k < cand.size(); ++k) {
                int j = cand[k];
                double d = 0;
                for (int i = 0; i < 192; ++i) d += qn[i] * knope[((size_t)j * NH + h) * 192 + i];
                for (int i = 0; i < 64; ++i) d += qp[i] * kpe[(size_t)j * 64 + i];
                sc[k] = d / 16.0;
                mx = std::max(mx, sc[k]);
            }
            double den = std::exp((double)W.sinks[h] - mx);
            for (auto& s : sc) { s = std::exp(s - mx); den += s; }
            double* o = &o_ref[((size_t)t * NH + h) * 256];
            for (size_t k = 0; k < cand.size(); ++k) {
                double p = sc[k] / den;
                const double* v = &vv[((size_t)cand[k] * NH + h) * 256];
                for (int i = 0; i < 256; ++i) o[i] += p * v[i];
            }
            if (in.gate) for (int i = 0; i < 256; ++i) o[i] *= 1.0 / (1.0 + std::exp(-(double)in.gate[(size_t)t * NH * 256 + (size_t)h * 256 + i]));
        }
    }
}

// ---- one test case ----
struct Case {
    const char* name;
    int mode, P, p0, M, kcount;  // kcount = keys per query in list mode
    bool gate;
    int chunk;                   // kv write chunk
};

static void to_half(const std::vector<float>& f, std::vector<uint16_t>& h) {
    h.resize(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = f2h(f[i]);
}

static bool run_case(const Weights& W, const Case& c, uint64_t seed, double tol, bool simple_only, bool dec = false, double qscale_in = -1) {
    const int NH = W.NH;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q_b((size_t)c.M * NH * 256), kv_a((size_t)c.P * 576), gate;
    // q: calibrate so that attention logits have std ~ 2.5 (flat-ish softmax would hide errors, spiky one too)
    for (auto& x : kv_a) x = nd(rng) * 2.5f;
    for (auto& x : q_b) x = nd(rng);
    if (c.gate) { gate.resize((size_t)c.M * NH * 256); for (auto& x : gate) x = 0.8f * nd(rng); }
    double qscale = qscale_in;
    if (qscale < 0) {
        // estimate logit std for query 0, head 0..3 against first 64 keys with scale 1
        double acc = 0; int n = 0;
        std::vector<double> kn(192);
        for (int j = 0; j < std::min(c.P, 64); ++j) {
            double ss = 0; for (int cc = 0; cc < 512; ++cc) ss += (double)kv_a[(size_t)j * 576 + cc] * kv_a[(size_t)j * 576 + cc];
            double inv = 1.0 / std::sqrt(ss / 512 + 1e-5);
            for (int h = 0; h < std::min(NH, 4); ++h) {
                for (int i = 0; i < 192; ++i) { double s = 0; for (int cc = 0; cc < 512; ++cc) s += (double)W.wk[((size_t)h * 192 + i) * 512 + cc] * kv_a[(size_t)j * 576 + cc] * inv * W.normw[cc]; kn[i] = s; }
                double d = 0; for (int i = 0; i < 192; ++i) d += (double)q_b[(size_t)h * 256 + i] * kn[i];
                acc += d * d / 256.0; ++n;
            }
        }
        double sd = std::sqrt(acc / n);
        qscale = sd > 1e-9 ? 2.5 / sd : 1.0;
    }
    // nope part scaled to logit std ~2.5; the rope part has a fixed N(0,1.2) so its logits (0.5*1.2*2.5 = 1.5 std) stay in the same range
    for (size_t i = 0; i < q_b.size(); ++i) q_b[i] *= (i % 256) < 192 ? (float)qscale : 1.2f;

    std::vector<int32_t> keys, nkeys;
    int kmax = 0;
    if (c.mode == MLA_LIST) {
        // IndexPool-like lists: random complete 4-token groups, then the (partial) group of the query itself last
        kmax = (c.kcount + 3) / 4 * 4;
        keys.assign((size_t)c.M * kmax, 0);
        nkeys.assign(c.M, 0);
        for (int t = 0; t < c.M; ++t) {
            int pos = c.p0 + t;
            int ncomplete = (pos + 1) / 4;  // groups 0..ncomplete-1 are complete (4G+3 <= pos); group pos/4 is the current one
            if (ncomplete * 4 + 3 > pos && ncomplete > 0 && ncomplete - 1 == pos / 4) --ncomplete;  // the current group is never counted complete
            std::vector<int> pool;
            for (int g = 0; g < pos / 4; ++g) pool.push_back(g);
            std::shuffle(pool.begin(), pool.end(), rng);
            int nsel = std::min<int>((int)pool.size(), kmax / 4 - 1);
            int n = 0;
            for (int i = 0; i < nsel; ++i)
                for (int u = 0; u < 4; ++u) keys[(size_t)t * kmax + n++] = pool[i] * 4 + u;
            for (int u = 0; u <= pos % 4; ++u) keys[(size_t)t * kmax + n++] = (pos / 4) * 4 + u;
            nkeys[t] = n;
            (void)ncomplete;
        }
    }
    RefIn rin{c.M, c.P, c.p0, c.mode, 512, q_b.data(), kv_a.data(), c.gate ? gate.data() : nullptr, &keys, &nkeys, kmax};
    std::vector<double> o_ref;
    ref_attention(W, rin, o_ref);

    // GPU
    std::vector<uint16_t> wk16, wv16;
    to_half(W.wk, wk16);
    to_half(W.wv, wv16);
    Dev<uint16_t> d_wk, d_wv;
    d_wk.up(wk16);
    d_wv.up(wv16);
    Dev<float> d_q, d_kv, d_norm, d_sink, d_gate;
    Dev<uint16_t> d_ctx;
    d_q.up(q_b); d_kv.up(kv_a); d_norm.up(W.normw); d_sink.up(W.sinks);
    if (c.gate) d_gate.up(gate);
    int ring = c.mode == MLA_WINDOW ? 512 + 64 : 0;
    int slots = ring ? ring : (c.P + 15) / 16 * 16;
    Dev<uint16_t> d_cache(mla_cache_bytes(slots) / 2);
    CK(hipMemset(d_cache.p, 0, d_cache.n * 2));
    Dev<uint16_t> d_qa((size_t)c.M * NH * 576), d_o((size_t)c.M * NH * 256);
    d_ctx.alloc((size_t)c.M * NH * 512);
    Dev<int32_t> d_keys, d_nkeys;
    if (c.mode == MLA_LIST) { d_keys.up(keys); d_nkeys.up(nkeys); }
    hipStream_t s = 0;
    for (int p = 0; p < c.P; p += c.chunk) {
        int m = std::min(c.chunk, c.P - p);
        mla_write_kv(s, d_kv.p + (size_t)p * 576, d_norm.p, (__half*)d_cache.p, slots, m, p, ring, (float)W.theta, 1e-5f);
    }
    Dev<uint16_t> d_wkt((size_t)NH * 512 * 192);
    mla_transpose_wk(s, (const __half*)d_wk.p, (__half*)d_wkt.p, NH);
    Dev<uint8_t> d_kvb;
    if (dec) { d_kvb.up(W.kvb_raw); mla_prep_q_dec(s, W.kvb_fmt, d_kvb.p, d_q.p, (__half*)d_qa.p, c.M, NH, c.p0, (float)W.theta); }
    else if (simple_only) mla_prep_q(s, d_q.p, (const __half*)d_wk.p, (__half*)d_qa.p, c.M, NH, c.p0, (float)W.theta);
    else mla_prep_q_t(s, d_q.p, (const __half*)d_wkt.p, (__half*)d_qa.p, c.M, NH, c.p0, (float)W.theta);
    MlaAttnArgs a;
    a.M = c.M; a.NH = NH; a.p0 = c.p0; a.mode = c.mode; a.window = 512; a.ring = ring; a.cache_slots = slots;
    a.keys = d_keys.p; a.nkeys = d_nkeys.p; a.kmax = kmax; a.scale = 0.0625f;
    g_mla_force_simple = simple_only ? 1 : 0;
    size_t wsb = mla_workspace_bytes(a);
    Dev<uint8_t> d_ws(wsb ? wsb : 16);
    if (dec) {
        mla_attend(s, (const __half*)d_qa.p, (const __half*)d_cache.p, d_sink.p, (__half*)d_ctx.p, a, d_ws.p, d_ws.n);
        mla_vup_dec(s, W.kvb_fmt, d_kvb.p, (const __half*)d_ctx.p, c.gate ? d_gate.p : nullptr, (__half*)d_o.p, c.M, NH);
    } else {
        mla_attention(s, (const __half*)d_qa.p, (const __half*)d_cache.p, d_sink.p, (const __half*)d_wv.p, c.gate ? d_gate.p : nullptr, (__half*)d_o.p, (float*)d_ctx.p, a, d_ws.p, d_ws.n);
    }
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> o16;
    d_o.down(o16);
    double mxr = 0, mxe = 0, se = 0, sr = 0;
    for (size_t i = 0; i < o16.size(); ++i) {
        double g = h2f(o16[i]), r = o_ref[i];
        mxr = std::max(mxr, std::fabs(r));
        mxe = std::max(mxe, std::fabs(g - r));
        se += (g - r) * (g - r);
        sr += r * r;
    }
    double rel = mxe / std::max(mxr, 1e-30), rrms = std::sqrt(se / std::max(sr, 1e-30));
    bool ok = rel < tol;
    printf("  [%s] %-34s NH=%2d P=%5d p0=%5d M=%4d  max|ref|=%.4f max_err=%.3e rel_max=%.3e rel_rms=%.3e  qscale=%.3g  %s\n", dec ? "dec   " : (simple_only ? "simple" : "fast  "), c.name, NH, c.P, c.p0, c.M, mxr, mxe, rel, rrms, qscale,
           ok ? "PASS" : "FAIL");
    return ok;
}

// ---- timings (no reference: cross-checked against the simple kernel on the same data) ----
static double time_ms(int reps, const std::function<void()>& fn) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    fn();  // warmup
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) fn();
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a);
    hipEventDestroy(b);
    return ms / reps;
}

struct PerfCfg { int NH, M, Tk, mode; const char* name; };

static void perf_case(const PerfCfg& pc, bool check_vs_simple) {
    const int NH = pc.NH, M = pc.M, Tk = pc.Tk;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    int ring = pc.mode == MLA_WINDOW ? 512 + 2048 : 0;
    int slots = ring ? ring : (Tk + 15) / 16 * 16;
    std::vector<uint16_t> cache(mla_cache_bytes(slots) / 2, 0);
    for (size_t i = 0; i < (size_t)slots * 576; ++i) cache[i] = f2h(ud(rng) * 1.5f);
    std::vector<uint16_t> qa((size_t)M * NH * 576), wv((size_t)NH * 256 * 512);
    for (auto& x : qa) x = f2h(ud(rng) * 0.5f);
    for (auto& x : wv) x = f2h(ud(rng) * 0.05f);
    std::vector<float> sinks(NH);
    for (auto& x : sinks) x = ud(rng);
    Dev<uint16_t> d_cache, d_qa, d_wv, d_o((size_t)M * NH * 256);
    d_cache.up(cache); d_qa.up(qa); d_wv.up(wv);
    Dev<float> d_sink;
    Dev<uint16_t> d_ctx((size_t)M * NH * 512);
    d_sink.up(sinks);
    MlaAttnArgs a;
    a.M = M; a.NH = NH; a.p0 = Tk - M; a.mode = pc.mode; a.window = 512; a.ring = ring; a.cache_slots = slots; a.scale = 0.0625f;
    Dev<int32_t> d_keys, d_nkeys;
    if (pc.mode == MLA_LIST) {  // IndexPool-like: 511 random complete groups + the own group = 2048 keys per query
        const int kmax = 2048;
        std::vector<int32_t> keys((size_t)M * kmax), nk(M, kmax);
        for (int t = 0; t < M; ++t) {
            for (int g = 0; g < 512; ++g) {
                int grp = (int)(rng() % (uint64_t)((Tk - M) / 4));
                for (int u = 0; u < 4; ++u) keys[(size_t)t * kmax + g * 4 + u] = grp * 4 + u;
            }
        }
        d_keys.up(keys); d_nkeys.up(nk);
        a.keys = d_keys.p; a.nkeys = d_nkeys.p; a.kmax = kmax;
    }
    size_t wsb = mla_workspace_bytes(a);
    Dev<uint8_t> d_ws(wsb ? wsb : 16);
    hipStream_t s = 0;
    g_mla_force_simple = 0;
    double keys_avg = pc.mode == MLA_LIST ? 2048.0 : (pc.mode == MLA_WINDOW ? std::min(512, Tk) : (double)(Tk - M) + (M + 1) / 2.0);
    if (pc.mode == MLA_CAUSAL && Tk == M) keys_avg = (M + 1) / 2.0;
    double ms_att = time_ms(M >= 64 ? 3 : 20, [&] { mla_attend(s, (const __half*)d_qa.p, (const __half*)d_cache.p, d_sink.p, (__half*)d_ctx.p, a, d_ws.p, d_ws.n); });
    g_mla_force_simple = 1;
    double ms_vup_s = time_ms(M >= 64 ? 3 : 20, [&] { mla_vup(s, (const __half*)d_ctx.p, (const __half*)d_wv.p, nullptr, (__half*)d_o.p, M, NH); });
    g_mla_force_simple = 0;
    double ms_vup = time_ms(M >= 64 ? 3 : 20, [&] { mla_vup(s, (const __half*)d_ctx.p, (const __half*)d_wv.p, nullptr, (__half*)d_o.p, M, NH); });
    printf("       vup simple %.3f ms\n", ms_vup_s);
    {
        std::vector<float> qb((size_t)M * NH * 256);
        for (auto& x : qb) x = ud(rng);
        std::vector<uint16_t> wk((size_t)NH * 192 * 512);
        for (auto& x : wk) x = f2h(ud(rng) * 0.05f);
        Dev<float> d_qb;
        d_qb.up(qb);
        Dev<uint16_t> d_wk, d_wkt((size_t)NH * 512 * 192), d_qa2((size_t)M * NH * 576);
        d_wk.up(wk);
        mla_transpose_wk(s, (const __half*)d_wk.p, (__half*)d_wkt.p, NH);
        double ms_ps = time_ms(M >= 64 ? 3 : 20, [&] { mla_prep_q(s, d_qb.p, (const __half*)d_wk.p, (__half*)d_qa2.p, M, NH, 0, 1e7f); });
        double ms_pw = time_ms(M >= 64 ? 3 : 20, [&] { mla_prep_q_t(s, d_qb.p, (const __half*)d_wkt.p, (__half*)d_qa2.p, M, NH, 0, 1e7f); });
        double fl = 2.0 * M * NH * 192 * 512;
        printf("       prep_q: simple %.3f ms (%.2f TFLOPs, wk_b read %.0f GB/s)  wmma %.3f ms (%.2f TFLOPs)\n", ms_ps, fl / (ms_ps * 1e-3) / 1e12,
               (double)NH * 192 * 512 * 2 / (ms_ps * 1e-3) / 1e9, ms_pw, fl / (ms_pw * 1e-3) / 1e12);
    }
    if (M <= 8) {
        std::vector<uint8_t> raw((size_t)NH * 448 * 2 * 166);
        for (auto& b : raw) b = (uint8_t)(rng() & 0xFF);
        for (size_t r = 0; r < (size_t)NH * 448 * 2; ++r) { raw[r * 166] = 0x00; raw[r * 166 + 1] = 0x28; }  // D = fp16 0x2800 (finite)
        Dev<uint8_t> d_kvb;
        d_kvb.up(raw);
        std::vector<float> qb((size_t)M * NH * 256);
        for (auto& x : qb) x = ud(rng);
        Dev<float> d_qb;
        d_qb.up(qb);
        Dev<uint16_t> d_qa3((size_t)M * NH * 576);
        double ms_pd = time_ms(50, [&] { mla_prep_q_dec(s, 5, d_kvb.p, d_qb.p, (__half*)d_qa3.p, M, NH, 0, 1e7f); });
        double ms_vd = time_ms(50, [&] { mla_vup_dec(s, 5, d_kvb.p, (const __half*)d_ctx.p, nullptr, (__half*)d_o.p, M, NH); });
        printf("       dec (S5_K kv_b): prep_q %.3f ms (%.0f GB/s of wk rows)  vup %.3f ms (%.0f GB/s of wv rows)\n", ms_pd, (double)NH * 192 * 332 / (ms_pd * 1e-3) / 1e9, ms_vd,
               (double)NH * 256 * 332 / (ms_vd * 1e-3) / 1e9);
    }
    double fl_att = 2.0 * M * NH * keys_avg * (576.0 + 512.0);
    double bytes_kv = keys_avg * 576 * 2.0;
    printf("  PERF %-22s NH=%2d M=%4d Tk=%6d  attend %.3f ms (%.2f TFLOPs, cache-row GB/s %.1f)  vup %.3f ms (%.2f TFLOPs)\n", pc.name, NH, M, Tk, ms_att, fl_att / (ms_att * 1e-3) / 1e12,
           bytes_kv * (M >= 64 ? 1 : 1) / (ms_att * 1e-3) / 1e9, ms_vup, 2.0 * M * NH * 256 * 512 / (ms_vup * 1e-3) / 1e12);
    if (M <= 8) {
        printf("       split sweep (target blocks -> attend ms):");
        const int save = g_mla_target_blocks;
        for (int tb : {32, 64, 128, 192, 384, 768}) {
            g_mla_target_blocks = tb;
            size_t w2 = mla_workspace_bytes(a);
            Dev<uint8_t> ws2(w2 ? w2 : 16);
            double ms = time_ms(20, [&] { mla_attend(s, (const __half*)d_qa.p, (const __half*)d_cache.p, d_sink.p, (__half*)d_ctx.p, a, ws2.p, ws2.n); });
            printf("  %d: %.3f", tb, ms);
        }
        printf("\n");
        g_mla_target_blocks = save;
    }
    if (check_vs_simple) {
        std::vector<uint16_t> ctx_fast16, ctx_simple16;
        d_ctx.down(ctx_fast16);
        g_mla_force_simple = 1;
        mla_attend(s, (const __half*)d_qa.p, (const __half*)d_cache.p, d_sink.p, (__half*)d_ctx.p, a, d_ws.p, d_ws.n);
        CK(hipDeviceSynchronize());
        d_ctx.down(ctx_simple16);
        g_mla_force_simple = 0;
        std::vector<float> ctx_fast(ctx_fast16.size()), ctx_simple(ctx_simple16.size());
        for (size_t i = 0; i < ctx_fast.size(); ++i) { ctx_fast[i] = h2f(ctx_fast16[i]); ctx_simple[i] = h2f(ctx_simple16[i]); }
        double mx = 0, me = 0;
        for (size_t i = 0; i < ctx_fast.size(); ++i) { mx = std::max(mx, (double)std::fabs(ctx_simple[i])); me = std::max(me, (double)std::fabs(ctx_fast[i] - ctx_simple[i])); }
        printf("       fast-vs-simple ctx: max|simple|=%.4f max_err=%.3e rel=%.3e  %s\n", mx, me, me / std::max(mx, 1e-30), me / std::max(mx, 1e-30) < 3e-3 ? "PASS" : "FAIL");
    }
}

int main(int argc, char** argv) {
    std::string model;
    bool perf = false, noref_real = false;
    std::vector<int> layers = {5, 2};
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--perf") perf = true;
        else if (a == "--quick") quick = true;
        else if (a == "--noreal") noref_real = true;
        else if (a == "--layers" && i + 1 < argc) { layers.clear(); std::string s = argv[++i]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); layers.push_back(atoi(s.substr(p, q - p).c_str())); if (q == std::string::npos) break; p = q + 1; } }
    }
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s %s CUs=%d\n", prop.name, gpu_arch(prop), prop.multiProcessorCount);
    int fails = 0;
    std::mt19937_64 rng(1234);
    const double TOL = 2e-3;
    for (int simple = 1; simple >= 0; --simple) {
        printf("== random weights (%s kernels) ==\n", simple ? "simple" : "fast");
        {
            Weights W;
            make_random_weights(W, 32, rng, 1e7);
            Case cs[] = {
                {"causal decode M=1 after 299", MLA_CAUSAL, 300, 299, 1, 0, true, 64},
                {"causal chunk M=40 after 260", MLA_CAUSAL, 300, 260, 40, 0, true, 64},
                {"causal chunk M=33 at 0", MLA_CAUSAL, 33, 0, 33, 0, false, 33},
                {"causal M=8 after 120 (no gate)", MLA_CAUSAL, 128, 120, 8, 0, false, 64},
                {"list M=12 k=70 after 300", MLA_LIST, 312, 300, 12, 70, true, 64},
                {"list M=3 k=300 (>256 tile)", MLA_LIST, 303, 300, 3, 300, true, 64},
            };
            for (auto& c : cs) fails += !run_case(W, c, rng(), TOL, simple);
        }
        {
            Weights W;
            make_random_weights(W, 64, rng, 1e4);
            Case cs[] = {
                {"window M=16 after 1100 (ring)", MLA_WINDOW, 1116, 1100, 16, 0, true, 64},
                {"window M=1 after 700", MLA_WINDOW, 701, 700, 1, 0, true, 64},
                {"window M=40 at 100 (short)", MLA_WINDOW, 140, 100, 40, 0, true, 64},
            };
            for (auto& c : cs) fails += !run_case(W, c, rng(), TOL, simple);
        }
        if (quick) break;
    }
    if (!model.empty() && !noref_real) {
        for (int L : layers) {
            printf("== real layer %d (%s) ==\n", L, model.c_str());
            Weights W;
            std::string err;
            if (!load_real_layer(W, model, L, err)) { printf("  load failed: %s\n", err.c_str()); ++fails; continue; }
            bool swa = W.NH == 64;
            for (int simple = 1; simple >= 0; --simple) {
                if (swa) {
                    Case c1{"real window M=16 after 700", MLA_WINDOW, 716, 700, 16, 0, true, 64};
                    Case c2{"real window M=1 after 900", MLA_WINDOW, 901, 900, 1, 0, true, 64};
                    fails += !run_case(W, c1, rng(), TOL, simple);
                    fails += !run_case(W, c2, rng(), TOL, simple);
                    if (!simple) fails += !run_case(W, c2, rng(), TOL, false, true);
                } else {
                    Case c1{"real causal M=24 after 300", MLA_CAUSAL, 324, 300, 24, 0, true, 64};
                    Case c2{"real causal M=1 after 600", MLA_CAUSAL, 601, 600, 1, 0, true, 64};
                    Case c3{"real list M=6 k=200 after 500", MLA_LIST, 506, 500, 6, 200, true, 64};
                    fails += !run_case(W, c1, rng(), TOL, simple);
                    fails += !run_case(W, c2, rng(), TOL, simple);
                    fails += !run_case(W, c3, rng(), TOL, simple);
                    if (!simple) { fails += !run_case(W, c2, rng(), TOL, false, true); fails += !run_case(W, c3, rng(), TOL, false, true); }
                }
                if (quick) break;
            }
        }
    }
    if (perf) {
        printf("== timings ==\n");
        PerfCfg decode[] = {
            {32, 1, 4096, MLA_CAUSAL, "decode sparse"},
            {32, 1, 8192, MLA_CAUSAL, "decode sparse"},
            {32, 1, 32768, MLA_CAUSAL, "decode sparse"},
            {64, 1, 512, MLA_WINDOW, "decode swa (window)"},
            {64, 1, 8192, MLA_WINDOW, "decode swa (window)"},
            {32, 8, 8192, MLA_CAUSAL, "verify M=8 sparse"},
            {32, 1, 32768, MLA_LIST, "decode list 2048 keys"},
            {32, 1, 131072, MLA_LIST, "decode list 2048 @128K"},
            {32, 8, 32768, MLA_LIST, "verify M=8 list 2048"},
        };
        for (auto& p : decode) perf_case(p, true);
        PerfCfg pre[] = {
            {64, 2048, 2048, MLA_WINDOW, "prefill swa window"},
            {64, 2048, 8192, MLA_WINDOW, "prefill swa window"},
            {32, 2048, 2048, MLA_CAUSAL, "prefill sparse causal"},
            {32, 2048, 8192, MLA_CAUSAL, "prefill sparse causal"},
            {32, 2048, 32768, MLA_CAUSAL, "prefill sparse causal"},
        };
        for (auto& p : pre) perf_case(p, false);
        PerfCfg chk[] = {{32, 256, 3000, MLA_CAUSAL, "check prefill causal"}, {64, 300, 2000, MLA_WINDOW, "check prefill window"}, {32, 128, 6000, MLA_CAUSAL, "check prefill causal"}};
        for (auto& p : chk) perf_case(p, true);
    }
    printf("%s (fails=%d)\n", fails ? "SOME TESTS FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
