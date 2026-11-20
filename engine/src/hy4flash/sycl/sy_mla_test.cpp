// sy_mla_test.cpp - correctness (vs an fp32/double non-absorbed CPU reference) and timing of the SYCL MLA module (sy_mla.h/.cpp)
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "sy_linear.h"
#include "sy_mla.h"

using namespace hy4s;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 2654435761ull + 88172645463325252ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return (float)((next() >> 11) * (1.0 / 9007199254740992.0)); }
    float nrm() {
        float u1 = std::max(uni(), 1e-12f), u2 = uni();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }
};

static float h2f(half h) { return (float)h; }

struct Scn {
    const char* name;
    int NH, mode, p0, M, ring, window, kmax;
    float theta;
};

struct Err {
    double max_abs = 0, max_ref = 0, cos = 1;
};
static Err cmp(const std::vector<float>& got, const std::vector<float>& ref) {
    Err e;
    double dot = 0, ng = 0, nr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        e.max_abs = std::max(e.max_abs, (double)std::fabs(got[i] - ref[i]));
        e.max_ref = std::max(e.max_ref, (double)std::fabs(ref[i]));
        dot += (double)got[i] * ref[i];
        ng += (double)got[i] * got[i];
        nr += (double)ref[i] * ref[i];
    }
    e.cos = dot / (std::sqrt(ng * nr) + 1e-30);
    return e;
}

static void rope_ref(double* x, int D, int pos, double theta) {
    for (int i = 0; i < D / 2; ++i) {
        double f = std::pow(theta, -(double)(2 * i) / D), a = pos * f, c = std::cos(a), s = std::sin(a);
        double x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

static bool run_scn(const Scn& sc, uint64_t seed) {
    Rng rng(seed);
    const int NH = sc.NH, M = sc.M, p0 = sc.p0, Ntot = p0 + M;
    const int slots = sc.ring > 0 ? sc.ring : ((Ntot + 15) / 16) * 16 + 16;
    const float eps = 1e-5f;
    // ---- data
    std::vector<float> kv_a((size_t)Ntot * MLA_ROW), norm_w(MLA_KV_LORA);
    for (auto& v : kv_a) v = rng.nrm();
    for (auto& v : norm_w) v = 0.5f + rng.uni();
    std::vector<float> q_b((size_t)M * NH * MLA_QHEAD), gate((size_t)M * NH * MLA_VHEAD), sinks(NH);
    for (auto& v : q_b) v = rng.nrm();
    for (auto& v : gate) v = rng.nrm();
    for (auto& v : sinks) v = rng.nrm();
    std::vector<half> wk((size_t)NH * MLA_NOPE * MLA_KV_LORA), wv((size_t)NH * MLA_VHEAD * MLA_KV_LORA);
    const float swk = 1.2f / std::sqrt(192.0f), swv = 1.0f / std::sqrt(512.0f);
    for (auto& v : wk) v = (half)(rng.nrm() * swk);
    for (auto& v : wv) v = (half)(rng.nrm() * swv);
    // list mode: random key lists (cache slots = positions), no duplicates, includes the query's own position
    std::vector<int32_t> keys, nkeys;
    if (sc.mode == MLA_LIST) {
        keys.assign((size_t)M * sc.kmax, 0);
        nkeys.assign(M, 0);
        for (int t = 0; t < M; ++t) {
            int pos = p0 + t;
            int n = std::min(pos + 1, sc.kmax - (int)(rng.next() % 37));
            std::vector<int> all(pos + 1);
            for (int i = 0; i <= pos; ++i) all[i] = i;
            for (int i = 0; i < n - 1 && i < pos; ++i) {  // partial shuffle of 0..pos-1, keep pos
                int j = i + (int)(rng.next() % (uint64_t)(pos - i));
                std::swap(all[i], all[j]);
            }
            int c = 0;
            for (int i = 0; i < n - 1 && i < pos; ++i) keys[(size_t)t * sc.kmax + c++] = all[i];
            keys[(size_t)t * sc.kmax + c++] = pos;
            nkeys[t] = c;
        }
    }
    // ---- device
    float *d_kv = dalloc<float>(kv_a.size()), *d_nw = dalloc<float>(MLA_KV_LORA), *d_qb = dalloc<float>(q_b.size()), *d_gate = dalloc<float>(gate.size()), *d_sink = dalloc<float>(NH);
    half *d_wk = dalloc<half>(wk.size()), *d_wv = dalloc<half>(wv.size());
    half *d_cache = dalloc<half>((size_t)slots * MLA_ROW), *d_qa = dalloc<half>((size_t)M * NH * MLA_ROW), *d_ctx = dalloc<half>((size_t)M * NH * MLA_KV_LORA),
         *d_o = dalloc<half>((size_t)M * NH * MLA_VHEAD);
    int32_t *d_keys = dalloc<int32_t>(std::max<size_t>(keys.size(), 1)), *d_nk = dalloc<int32_t>(std::max<size_t>(nkeys.size(), 1));
    h2d(d_kv, kv_a.data(), kv_a.size() * 4);
    h2d(d_nw, norm_w.data(), MLA_KV_LORA * 4);
    h2d(d_qb, q_b.data(), q_b.size() * 4);
    h2d(d_gate, gate.data(), gate.size() * 4);
    h2d(d_sink, sinks.data(), NH * 4);
    h2d(d_wk, wk.data(), wk.size() * 2);
    h2d(d_wv, wv.data(), wv.size() * 2);
    if (!keys.empty()) {
        h2d(d_keys, keys.data(), keys.size() * 4);
        h2d(d_nk, nkeys.data(), nkeys.size() * 4);
    }
    dzero(d_cache, (size_t)slots * MLA_ROW * 2);
    // write the cache in chunks of 48 tokens (like prefill chunks)
    for (int pc = 0; pc < Ntot; pc += 48) {
        int n = std::min(48, Ntot - pc);
        mla_write_kv(d_kv + (size_t)pc * MLA_ROW, d_nw, d_cache, slots, n, pc, sc.ring, sc.theta, eps);
    }
    mla_prep_q(d_qb, d_wk, d_qa, M, NH, p0, sc.theta);
    MlaAttnArgs a;
    a.M = M; a.NH = NH; a.p0 = p0; a.mode = sc.mode; a.window = sc.window; a.ring = sc.ring; a.cache_slots = slots;
    a.keys = d_keys; a.nkeys = d_nk; a.kmax = sc.kmax;
    size_t wsb = mla_workspace_bytes(a);
    void* d_ws = wsb ? dalloc<uint8_t>(wsb) : nullptr;
    mla_attend(d_qa, d_cache, d_sink, d_ctx, a, d_ws, wsb);
    mla_vup(d_ctx, d_wv, d_gate, d_o, M, NH);
    wait();
    std::vector<half> h_cache((size_t)slots * MLA_ROW), h_qa((size_t)M * NH * MLA_ROW), h_o((size_t)M * NH * MLA_VHEAD);
    d2h(h_cache.data(), d_cache, h_cache.size() * 2);
    d2h(h_qa.data(), d_qa, h_qa.size() * 2);
    d2h(h_o.data(), d_o, h_o.size() * 2);
    wait();
    // ---- reference
    std::vector<float> lat((size_t)Ntot * MLA_KV_LORA), kpe((size_t)Ntot * MLA_ROPE);
    for (int p = 0; p < Ntot; ++p) {
        const float* r = &kv_a[(size_t)p * MLA_ROW];
        double ss = 0;
        for (int c = 0; c < MLA_KV_LORA; ++c) ss += (double)r[c] * r[c];
        double inv = 1.0 / std::sqrt(ss / MLA_KV_LORA + (double)eps);
        for (int c = 0; c < MLA_KV_LORA; ++c) lat[(size_t)p * MLA_KV_LORA + c] = (float)(r[c] * inv * norm_w[c]);
        double pe[64];
        for (int i = 0; i < 64; ++i) pe[i] = r[MLA_KV_LORA + i];
        rope_ref(pe, 64, p, sc.theta);
        for (int i = 0; i < 64; ++i) kpe[(size_t)p * 64 + i] = (float)pe[i];
    }
    // cache check (rows that are live: for ring, the last `ring` positions)
    std::vector<float> cache_got, cache_ref;
    for (int p = std::max(0, Ntot - (sc.ring > 0 ? sc.ring : Ntot)); p < Ntot; ++p) {
        int slot = sc.ring > 0 ? p % sc.ring : p;
        for (int c = 0; c < MLA_KV_LORA; ++c) { cache_got.push_back(h2f(h_cache[(size_t)slot * MLA_ROW + c])); cache_ref.push_back(lat[(size_t)p * MLA_KV_LORA + c]); }
        for (int c = 0; c < 64; ++c) { cache_got.push_back(h2f(h_cache[(size_t)slot * MLA_ROW + MLA_KV_LORA + c])); cache_ref.push_back(kpe[(size_t)p * 64 + c]); }
    }
    Err e_cache = cmp(cache_got, cache_ref);
    // per head: k_nope = wk_b * lat, v = wv_b * lat for the positions that can be seen
    int pmin = 0;
    if (sc.mode == MLA_WINDOW) pmin = std::max(0, p0 - sc.window + 1);
    const int np = Ntot - pmin;
    std::vector<float> ref_o((size_t)M * NH * MLA_VHEAD), ref_qa((size_t)M * NH * MLA_ROW);
    std::vector<float> wkf(wk.size()), wvf(wv.size());
    for (size_t i = 0; i < wk.size(); ++i) wkf[i] = h2f(wk[i]);
    for (size_t i = 0; i < wv.size(); ++i) wvf[i] = h2f(wv[i]);
#pragma omp parallel for
    for (int h = 0; h < NH; ++h) {
        std::vector<float> kn((size_t)np * MLA_NOPE), vv((size_t)np * MLA_VHEAD);
        for (int pi = 0; pi < np; ++pi) {
            const float* l = &lat[(size_t)(pmin + pi) * MLA_KV_LORA];
            for (int i = 0; i < MLA_NOPE; ++i) {
                const float* w = &wkf[((size_t)h * MLA_NOPE + i) * MLA_KV_LORA];
                float s = 0;
                for (int c = 0; c < MLA_KV_LORA; ++c) s += w[c] * l[c];
                kn[(size_t)pi * MLA_NOPE + i] = s;
            }
            for (int j = 0; j < MLA_VHEAD; ++j) {
                const float* w = &wvf[((size_t)h * MLA_VHEAD + j) * MLA_KV_LORA];
                float s = 0;
                for (int c = 0; c < MLA_KV_LORA; ++c) s += w[c] * l[c];
                vv[(size_t)pi * MLA_VHEAD + j] = s;
            }
        }
        for (int t = 0; t < M; ++t) {
            const int pos = p0 + t;
            const float* qh = &q_b[((size_t)t * NH + h) * MLA_QHEAD];
            double qpe[64];
            for (int i = 0; i < 64; ++i) qpe[i] = qh[MLA_NOPE + i];
            rope_ref(qpe, 64, pos, sc.theta);
            // reference of the absorbed query (for the prep_q check)
            for (int c = 0; c < MLA_KV_LORA; ++c) {
                double s = 0;
                for (int i = 0; i < MLA_NOPE; ++i) s += (double)qh[i] * wkf[((size_t)h * MLA_NOPE + i) * MLA_KV_LORA + c];
                ref_qa[((size_t)t * NH + h) * MLA_ROW + c] = (float)s;
            }
            for (int i = 0; i < 64; ++i) ref_qa[((size_t)t * NH + h) * MLA_ROW + MLA_KV_LORA + i] = (float)qpe[i];
            std::vector<int> ks;
            if (sc.mode == MLA_CAUSAL) for (int j = 0; j <= pos; ++j) ks.push_back(j);
            else if (sc.mode == MLA_WINDOW) for (int j = std::max(0, pos - sc.window + 1); j <= pos; ++j) ks.push_back(j);
            else for (int i = 0; i < nkeys[t]; ++i) ks.push_back(keys[(size_t)t * sc.kmax + i]);
            std::vector<double> s(ks.size());
            double mx = sinks[h];
            for (size_t k = 0; k < ks.size(); ++k) {
                const int pi = ks[k] - pmin;
                double d = 0;
                for (int i = 0; i < MLA_NOPE; ++i) d += (double)qh[i] * kn[(size_t)pi * MLA_NOPE + i];
                for (int i = 0; i < 64; ++i) d += qpe[i] * kpe[(size_t)ks[k] * 64 + i];
                s[k] = d / 16.0;
                mx = std::max(mx, s[k]);
            }
            double den = std::exp((double)sinks[h] - mx);
            for (size_t k = 0; k < ks.size(); ++k) { s[k] = std::exp(s[k] - mx); den += s[k]; }
            for (int j = 0; j < MLA_VHEAD; ++j) {
                double o = 0;
                for (size_t k = 0; k < ks.size(); ++k) o += s[k] * vv[(size_t)(ks[k] - pmin) * MLA_VHEAD + j];
                double g = gate[(size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + j];
                ref_o[(size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + j] = (float)(o / den / (1.0 + std::exp(-g)));
            }
        }
    }
    std::vector<float> got_qa(h_qa.size()), got_o(h_o.size());
    for (size_t i = 0; i < h_qa.size(); ++i) got_qa[i] = h2f(h_qa[i]);
    for (size_t i = 0; i < h_o.size(); ++i) got_o[i] = h2f(h_o[i]);
    Err e_qa = cmp(got_qa, ref_qa), e_o = cmp(got_o, ref_o);
    bool ok = e_o.cos > 0.99999 && e_o.max_abs < 2e-3 * e_o.max_ref && e_qa.cos > 0.99999 && e_cache.cos > 0.99999;
    printf("%-34s NH=%2d M=%3d p0=%5d S_ws=%8.2f MB | cache cos %.7f maxabs %.2e | qa cos %.7f maxabs %.2e (ref %.2f) | o cos %.7f maxabs %.2e rel %.2e (ref %.3f) %s\n", sc.name, NH, M, p0,
           wsb / 1e6, e_cache.cos, e_cache.max_abs, e_qa.cos, e_qa.max_abs, e_qa.max_ref, e_o.cos, e_o.max_abs, e_o.max_abs / e_o.max_ref, e_o.max_ref, ok ? "OK" : "FAIL");
    for (void* p : {(void*)d_kv, (void*)d_nw, (void*)d_qb, (void*)d_gate, (void*)d_sink, (void*)d_wk, (void*)d_wv, (void*)d_cache, (void*)d_qa, (void*)d_ctx, (void*)d_o, (void*)d_keys, (void*)d_nk, d_ws})
        if (p) dfree(p);
    return ok;
}

// ---- timing
struct Perf {
    half *cache, *qa, *ctx, *o, *wk, *wv;
    float *qb, *gate, *sink;
    int32_t *keys, *nk;
    void* ws;
    size_t wsb;
    int slots;
};

static void time_case(const char* name, int NH, int mode, int M, int p0, int ring, int window, int kmax, int ctx_slots, int nk_list, int reps) {
    Rng rng(777);
    const int slots = ring > 0 ? ring : ctx_slots;
    std::vector<half> hc((size_t)slots * MLA_ROW);
    for (auto& v : hc) v = (half)(rng.nrm() * 0.5f);
    Perf P;
    P.cache = dalloc<half>(hc.size());
    h2d(P.cache, hc.data(), hc.size() * 2);
    P.qa = dalloc<half>((size_t)M * NH * MLA_ROW);
    std::vector<half> hq((size_t)M * NH * MLA_ROW);
    for (auto& v : hq) v = (half)(rng.nrm() * 0.3f);
    h2d(P.qa, hq.data(), hq.size() * 2);
    P.ctx = dalloc<half>((size_t)M * NH * MLA_KV_LORA);
    P.o = dalloc<half>((size_t)M * NH * MLA_VHEAD);
    P.wk = dalloc<half>((size_t)NH * MLA_NOPE * MLA_KV_LORA);
    P.wv = dalloc<half>((size_t)NH * MLA_VHEAD * MLA_KV_LORA);
    {
        std::vector<half> w((size_t)NH * MLA_NOPE * MLA_KV_LORA, (half)0.01f);
        h2d(P.wk, w.data(), w.size() * 2);
        std::vector<half> w2((size_t)NH * MLA_VHEAD * MLA_KV_LORA, (half)0.01f);
        h2d(P.wv, w2.data(), w2.size() * 2);
    }
    P.qb = dalloc<float>((size_t)M * NH * MLA_QHEAD);
    std::vector<float> hqb((size_t)M * NH * MLA_QHEAD);
    for (auto& v : hqb) v = rng.nrm();
    h2d(P.qb, hqb.data(), hqb.size() * 4);
    P.gate = dalloc<float>((size_t)M * NH * MLA_VHEAD);
    dzero(P.gate, (size_t)M * NH * MLA_VHEAD * 4);
    P.sink = dalloc<float>(NH);
    dzero(P.sink, NH * 4);
    std::vector<int32_t> keys, nkeys;
    if (mode == MLA_LIST) {
        keys.assign((size_t)M * kmax, 0);
        nkeys.assign(M, nk_list);
        for (int t = 0; t < M; ++t)
            for (int i = 0; i < nk_list; ++i) keys[(size_t)t * kmax + i] = (int)(rng.next() % (uint64_t)slots);
    }
    P.keys = dalloc<int32_t>(std::max<size_t>(keys.size(), 1));
    P.nk = dalloc<int32_t>(std::max<size_t>(nkeys.size(), 1));
    if (!keys.empty()) {
        h2d(P.keys, keys.data(), keys.size() * 4);
        h2d(P.nk, nkeys.data(), nkeys.size() * 4);
    }
    MlaAttnArgs a;
    a.M = M; a.NH = NH; a.p0 = p0; a.mode = mode; a.window = window; a.ring = ring; a.cache_slots = slots; a.keys = P.keys; a.nkeys = P.nk; a.kmax = kmax;
    P.wsb = mla_workspace_bytes(a);
    P.ws = P.wsb ? dalloc<uint8_t>(P.wsb) : nullptr;
    auto once = [&](int what) {  // 0 all, 1 attend only, 2 prep only, 3 vup only
        if (what == 0 || what == 2) mla_prep_q(P.qb, P.wk, P.qa, M, NH, p0, 1e4f);
        if (what == 0 || what == 1) mla_attend(P.qa, P.cache, P.sink, P.ctx, a, P.ws, P.wsb);
        if (what == 0 || what == 3) mla_vup(P.ctx, P.wv, P.gate, P.o, M, NH);
    };
    double res[4];
    for (int what = 0; what < 4; ++what) {
        for (int i = 0; i < 3; ++i) once(what);
        wait();
        double t0 = now_s();
        for (int i = 0; i < reps; ++i) once(what);
        wait();
        res[what] = (now_s() - t0) / reps * 1000.0;
    }
    printf("%-34s NH=%2d M=%3d | all %8.3f ms | attend %8.3f | prep_q %8.3f | vup %8.3f (ws %.1f MB)\n", name, NH, M, res[0], res[1], res[2], res[3], P.wsb / 1e6);
    for (void* p : {(void*)P.cache, (void*)P.qa, (void*)P.ctx, (void*)P.o, (void*)P.wk, (void*)P.wv, (void*)P.qb, (void*)P.gate, (void*)P.sink, (void*)P.keys, (void*)P.nk, P.ws})
        if (p) dfree(p);
}

int main(int argc, char** argv) {
    bool do_perf = true, do_corr = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-perf") do_perf = false;
        if (std::string(argv[i]) == "--no-corr") do_corr = false;
        if (std::string(argv[i]) == "--target" && i + 1 < argc) g_mla_target_blocks = atoi(argv[++i]);
    }
    printf("device %s | target blocks %d\n", Q().get_device().get_info<sycl::info::device::name>().c_str(), g_mla_target_blocks);
    {  // launch-overhead floor: one tiny kernel (write_kv, M=1) per call
        float* kv = dalloc<float>(MLA_ROW);
        float* nw = dalloc<float>(MLA_KV_LORA);
        half* c = dalloc<half>(32 * MLA_ROW);
        dzero(kv, MLA_ROW * 4);
        dzero(nw, MLA_KV_LORA * 4);
        for (int i = 0; i < 5; ++i) mla_write_kv(kv, nw, c, 32, 1, 0, 0, 1e4f, 1e-5f);
        wait();
        double t0 = now_s();
        for (int i = 0; i < 200; ++i) mla_write_kv(kv, nw, c, 32, 1, 0, 0, 1e4f, 1e-5f);
        double t1 = now_s();
        wait();
        printf("launch floor (write_kv M=1): %.1f us per call, of which host submit %.1f us\n", (now_s() - t0) / 200 * 1e6, (t1 - t0) / 200 * 1e6);
        dfree(kv); dfree(nw); dfree(c);
    }
    bool all = true;
    if (do_corr) {
        const Scn scs[] = {
            {"causal NH32 decode M=1 p0=300", 32, MLA_CAUSAL, 300, 1, 0, 512, 0, 1e7f},
            {"causal NH32 chunk M=37 p0=100", 32, MLA_CAUSAL, 100, 37, 0, 512, 0, 1e7f},
            {"causal NH64 chunk M=37 p0=100", 64, MLA_CAUSAL, 100, 37, 0, 512, 0, 1e4f},
            {"causal NH32 long p0=3000 M=1", 32, MLA_CAUSAL, 3000, 1, 0, 512, 0, 1e7f},
            {"window NH64 ring528 p0=1000 M=16", 64, MLA_WINDOW, 1000, 16, 528, 512, 0, 1e4f},
            {"window NH64 ring528 p0=700 M=1", 64, MLA_WINDOW, 700, 1, 528, 512, 0, 1e4f},
            {"window NH64 ring528 p0=100 M=37", 64, MLA_WINDOW, 100, 37, 1024, 512, 0, 1e4f},
            {"list NH32 kmax1024 p0=1500 M=1", 32, MLA_LIST, 1500, 1, 0, 512, 1024, 1e7f},
            {"list NH32 kmax1024 p0=1500 M=9", 32, MLA_LIST, 1500, 9, 0, 512, 1024, 1e7f},
            {"list NH64 kmax256 p0=900 M=5", 64, MLA_LIST, 900, 5, 0, 512, 256, 1e7f},
        };
        for (const auto& s : scs) all &= run_scn(s, 12345 + s.p0 + s.M);
        // Quantized S_K kv_b (mla_prep_q_sk / mla_vup_sk) vs split_kvb -> fp16 -> mla_prep_q / mla_vup
        {
            Rng rng(999);
            const int NH = 32, M = 4, p0 = 123;
            for (int fmt : {FMT_S4, FMT_S5, FMT_S8}) {
                const int sb = sk_sb_bytes(fmt);
                std::vector<uint8_t> w((size_t)NH * 448 * 2 * sb);
                for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)(rng.next() & 255);
                for (size_t i = 0; i < w.size(); i += sb) {
                    uint16_t d = sycl::bit_cast<uint16_t>((half)(0.01f + 0.02f * rng.uni()));
                    w[i] = d & 255; w[i + 1] = d >> 8;
                    for (int g = 0; g < 4; ++g) w[i + 2 + g] = (uint8_t)(100 + (rng.next() % 150));
                }
                std::vector<float> qb((size_t)M * NH * MLA_QHEAD), gt((size_t)M * NH * MLA_VHEAD);
                std::vector<half> ctx((size_t)M * NH * MLA_KV_LORA);
                for (auto& v : qb) v = rng.nrm();
                for (auto& v : gt) v = rng.nrm();
                for (auto& v : ctx) v = (half)(rng.nrm() * 0.3f);
                uint8_t* dw = dalloc<uint8_t>(w.size());
                half *dwk = dalloc<half>((size_t)NH * 192 * 512), *dwv = dalloc<half>((size_t)NH * 256 * 512);
                float *dqb = dalloc<float>(qb.size()), *dgt = dalloc<float>(gt.size());
                half *dctx = dalloc<half>(ctx.size()), *qa1 = dalloc<half>((size_t)M * NH * MLA_ROW), *qa2 = dalloc<half>((size_t)M * NH * MLA_ROW);
                half *o1 = dalloc<half>((size_t)M * NH * MLA_VHEAD), *o2 = dalloc<half>((size_t)M * NH * MLA_VHEAD);
                h2d(dw, w.data(), w.size()); h2d(dqb, qb.data(), qb.size() * 4); h2d(dgt, gt.data(), gt.size() * 4); h2d(dctx, ctx.data(), ctx.size() * 2);
                split_kvb(fmt, dw, dwk, dwv, NH);
                mla_prep_q(dqb, dwk, qa1, M, NH, p0, 1e7f);
                mla_prep_q_sk(fmt, dw, dqb, qa2, M, NH, p0, nullptr, false, 1e7f);
                mla_vup(dctx, dwv, dgt, o1, M, NH);
                mla_vup_sk(fmt, dw, dctx, dgt, o2, M, NH);
                std::vector<half> hqa1((size_t)M * NH * MLA_ROW), hqa2((size_t)M * NH * MLA_ROW), ho1((size_t)M * NH * MLA_VHEAD), ho2((size_t)M * NH * MLA_VHEAD);
                d2h(hqa1.data(), qa1, hqa1.size() * 2); d2h(hqa2.data(), qa2, hqa2.size() * 2);
                d2h(ho1.data(), o1, ho1.size() * 2); d2h(ho2.data(), o2, ho2.size() * 2);
                wait();
                std::vector<float> fqa1(hqa1.size()), fqa2(hqa2.size()), fo1(ho1.size()), fo2(ho2.size());
                for (size_t i = 0; i < hqa1.size(); ++i) { fqa1[i] = (float)hqa1[i]; fqa2[i] = (float)hqa2[i]; }
                for (size_t i = 0; i < ho1.size(); ++i) { fo1[i] = (float)ho1[i]; fo2[i] = (float)ho2[i]; }
                Err eq = cmp(fqa2, fqa1), eo = cmp(fo2, fo1);
                bool ok = eq.cos > 0.999999 && eo.cos > 0.999999 && eq.max_abs < 5e-3 * eq.max_ref && eo.max_abs < 5e-3 * eo.max_ref;
                printf("quant_kvb S%d M=%d: prep_q cos %.7f maxabs %.2e | vup cos %.7f maxabs %.2e %s\n",
                       fmt, M, eq.cos, eq.max_abs, eo.cos, eo.max_abs, ok ? "OK" : "FAIL");
                all &= ok;
                for (void* p : {(void*)dw, (void*)dwk, (void*)dwv, (void*)dqb, (void*)dgt, (void*)dctx, (void*)qa1, (void*)qa2, (void*)o1, (void*)o2}) dfree(p);
            }
        }
        printf("CORRECTNESS: %s\n", all ? "ALL OK" : "FAILED");
    }
    if (do_perf) {
        printf("--- decode M=1 (host time per call, in-order queue, %d reps)\n", 100);
        for (int ctx : {2048, 8192, 32768}) {
            char nm[64];
            snprintf(nm, sizeof nm, "LIST ctx %d", ctx);
            time_case(nm, 32, MLA_LIST, 1, ctx - 1, 0, 512, 2052, ctx + 16, std::min(ctx, 2052), 100);
            snprintf(nm, sizeof nm, "SWA window ctx %d", ctx);
            time_case(nm, 64, MLA_WINDOW, 1, ctx - 1, 528, 512, 0, 528, 0, 100);
            snprintf(nm, sizeof nm, "CAUSAL ctx %d", ctx);
            time_case(nm, 32, MLA_CAUSAL, 1, ctx - 1, 0, 512, 0, ctx + 16, 0, 100);
        }
        printf("--- decode M=4 (verify window)\n");
        time_case("LIST ctx 8192 M=4", 32, MLA_LIST, 4, 8188, 0, 512, 2052, 8208, 2052, 100);
        time_case("SWA ctx 8192 M=4", 64, MLA_WINDOW, 4, 8188, 528, 512, 0, 528, 0, 100);
        printf("--- prefill chunk M=512\n");
        time_case("CAUSAL p0=0", 32, MLA_CAUSAL, 512, 0, 0, 512, 0, 1040, 0, 5);
        time_case("CAUSAL p0=1536 (to 2048)", 32, MLA_CAUSAL, 512, 1536, 0, 512, 0, 2064, 0, 5);
        time_case("SWA window p0=2048 ring1040", 64, MLA_WINDOW, 512, 2048, 1040, 512, 0, 1040, 0, 5);
        time_case("LIST p0=4096 kmax2052", 32, MLA_LIST, 512, 4096, 0, 512, 2052, 4624, 2052, 5);
    }
    return all ? 0 : 1;
}
