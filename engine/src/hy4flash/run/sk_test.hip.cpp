// sk_test.hip.cpp - correctness + performance tests of skgemm.hip.h against the CPU reference (sk_codecs.cpp) and real SQ-64 tensors
#include "skgemm.hip.h"
#ifdef SK_FAST
#include "skhy4.hip.h"
#endif
#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace hy4;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static float rndf() { return (rnd() & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f; }

static int g_fail = 0, g_pass = 0;

static float half_to_float_host(__half h) { return __half2float(h); }

// CPU dequant of a [N,K] weight in wire format to float
static void cpu_dequant(int fmt, const uint8_t* raw, int N, int K, std::vector<float>& W) {
    W.assign((size_t)N * K, 0.f);
    if (fmt == FMT_Q8ROW) {
        const float* sc = (const float*)raw;
        const int8_t* q = (const int8_t*)(raw + (size_t)N * 4);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) W[(size_t)n * K + k] = (float)q[(size_t)n * K + k] * sc[n];
        return;
    }
    const size_t rb = (size_t)(K / 256) * sk_sb_bytes(fmt);
    for (int n = 0; n < N; ++n)
        strata::quant::sk::dequant_sk_row(raw + n * rb, (strata::quant::sk::SkCode)fmt, W.data() + (size_t)n * K, K);
}

struct Dev {
    void* p = nullptr;
    Dev(size_t b) { CK(hipMalloc(&p, b ? b : 16)); }
    ~Dev() { if (p) hipFree(p); }
};

// returns max abs error / max abs value
static double test_linear(const char* name, int fmt, const uint8_t* raw, size_t raw_bytes, int N, int K, const std::vector<int>& Ms) {
    std::vector<float> W;
    cpu_dequant(fmt, raw, N, K, W);
    Dev dW(raw_bytes + 256);  // the WMMA loaders may read a few bytes past the last super-block
    CK(hipMemcpy(dW.p, raw, raw_bytes, hipMemcpyHostToDevice));
    double worst = 0;
    for (int M : Ms) {
        std::vector<__half> X((size_t)M * K);
        for (auto& x : X) x = __float2half(rndf());
        Dev dX((size_t)M * K * 2), dY((size_t)M * N * 4);
        CK(hipMemcpy(dX.p, X.data(), X.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemset(dY.p, 0xFF, (size_t)M * N * 4));
#ifdef SK_FAST
        linear_h(0, fmt, dW.p, N, K, (const __half*)dX.p, (float*)dY.p, M);  // production dispatcher (GEMV for M <= 16, WMMA GEMM above)
#else
        linear_h_ref(0, fmt, dW.p, N, K, (const __half*)dX.p, (float*)dY.p, M);
#endif
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        std::vector<float> Y((size_t)M * N);
        CK(hipMemcpy(Y.data(), dY.p, Y.size() * 4, hipMemcpyDeviceToHost));
        double maxv = 0, maxe = 0;
        std::vector<double> ref((size_t)M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double a = 0;
                const float* wr = &W[(size_t)n * K];
                const __half* xr = &X[(size_t)m * K];
                for (int k = 0; k < K; ++k) a += (double)wr[k] * (double)half_to_float_host(xr[k]);
                ref[(size_t)m * N + n] = a;
                maxv = std::max(maxv, std::fabs(a));
            }
        for (size_t i = 0; i < ref.size(); ++i) {
            double e = std::fabs(ref[i] - (double)Y[i]);
            if (!(e == e)) e = 1e30;
            maxe = std::max(maxe, e);
        }
        const double rel = maxv > 0 ? maxe / maxv : maxe;
        worst = std::max(worst, rel);
        const bool ok = rel < 2e-3;
        uint64_t hh = 1469598103934665603ull;
        for (size_t i = 0; i < Y.size() * 4; ++i) { hh ^= ((const uint8_t*)Y.data())[i]; hh *= 1099511628211ull; }
        printf("  %-52s fmt %3d N=%5d K=%5d M=%3d  max|y|=%.4g  rel_err=%.3g  hash=%016llx %s\n", name, fmt, N, K, M, maxv, rel, (unsigned long long)hh, ok ? "ok" : "FAIL");
        if (ok) ++g_pass; else ++g_fail;
    }
    return worst;
}

static void test_random_codes() {
    printf("== random super-blocks vs CPU dequant_sk_row\n");
#ifdef SK_OLD
    for (int fmt : {FMT_S2, FMT_S3, FMT_S4, FMT_S5, FMT_S6, FMT_S8}) {
#else
    for (int fmt : {FMT_S1_25, FMT_S1_5, FMT_S2, FMT_S3, FMT_S4, FMT_S5, FMT_S6, FMT_S8}) {
#endif
        g_rng = 0x9E3779B97F4A7C15ull ^ (0x1000193ull * (uint64_t)(fmt + 1));  // per-code data: identical in old/new builds
        const int N = 96, K = 1536;
        const size_t sb = sk_sb_bytes(fmt);
        std::vector<uint8_t> raw((size_t)N * (K / 256) * sb);
        for (size_t b = 0; b < raw.size() / sb; ++b) {
            uint8_t* blk = &raw[b * sb];
            for (size_t i = 0; i < sb; ++i) blk[i] = (uint8_t)rnd();
            const float D = (0.001f + (rnd() & 1023) / 1024.0f * 0.05f);
            const __half h = __float2half(D);
            std::memcpy(blk, &h, 2);
            if (fmt == FMT_S8) for (int i = 0; i < 256; ++i) if ((int8_t)blk[6 + i] == -128) blk[6 + i] = 0;
        }
        char nm[64];
        snprintf(nm, sizeof nm, "random %s", fmt == FMT_S1_25 ? "S1.25" : fmt == FMT_S1_5 ? "S1.5" : fmt == FMT_S8 ? "S8_K" : fmt == FMT_S6 ? "S6_K" : fmt == FMT_S5 ? "S5_K"
                                                : fmt == FMT_S4 ? "S4_K" : fmt == FMT_S3 ? "S3_K" : "S2_K");
        test_linear(nm, fmt, raw.data(), raw.size(), N, K, {1, 3, 5, 8, 12, 16, 100});
        // embed_rows on the same rows
        {
            std::vector<int32_t> ids = {0, 7, 95, 50};
            std::vector<float> cpu(ids.size() * (size_t)K);
            const size_t rb = (size_t)(K / 256) * sb;
            for (size_t i = 0; i < ids.size(); ++i)
                strata::quant::sk::dequant_sk_row(raw.data() + (size_t)ids[i] * rb, (strata::quant::sk::SkCode)fmt, cpu.data() + i * K, K);
            Dev dW(raw.size() + 256), dI(ids.size() * 4), dO(cpu.size() * 4);
            CK(hipMemcpy(dW.p, raw.data(), raw.size(), hipMemcpyHostToDevice));
            CK(hipMemcpy(dI.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
            embed_rows(0, fmt, dW.p, N, K, (const int32_t*)dI.p, (float*)dO.p, (int)ids.size());
            CK(hipDeviceSynchronize());
            std::vector<float> got(cpu.size());
            CK(hipMemcpy(got.data(), dO.p, got.size() * 4, hipMemcpyDeviceToHost));
            double maxe = 0, maxv = 0;
            for (size_t i = 0; i < got.size(); ++i) { maxe = std::max(maxe, (double)std::fabs(got[i] - cpu[i])); maxv = std::max(maxv, (double)std::fabs(cpu[i])); }
            const bool ok = maxe <= 1e-6 * maxv;
            printf("  %-52s embed_rows max abs err %.3g (max |w| %.3g)  %s\n", nm, maxe, maxv, ok ? "ok" : "FAIL");
            if (ok) ++g_pass; else ++g_fail;
        }
    }
    // Q8ROW random, K not a multiple of 256
    {
        const int N = 50, K = 128;
        std::vector<uint8_t> raw((size_t)N * 4 + (size_t)N * K);
        float* sc = (float*)raw.data();
        for (int n = 0; n < N; ++n) sc[n] = 0.001f + (rnd() & 255) / 25600.f;
        int8_t* q = (int8_t*)(raw.data() + N * 4);
        for (int i = 0; i < N * K; ++i) q[i] = (int8_t)((int)(rnd() % 255) - 127);
        test_linear("random Q8ROW", FMT_Q8ROW, raw.data(), raw.size(), N, K, {1, 3, 8, 16, 100});
    }
    // linear_f32
    {
        const int N = 512, K = 3072;
        std::vector<float> W((size_t)N * K);
        for (auto& w : W) w = rndf() * 0.05f;
        for (int M : {1, 3, 8, 16, 100}) {
            std::vector<float> X((size_t)M * K);
            for (auto& x : X) x = rndf();
            Dev dW(W.size() * 4), dX(X.size() * 4), dY((size_t)M * N * 4);
            CK(hipMemcpy(dW.p, W.data(), W.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dX.p, X.data(), X.size() * 4, hipMemcpyHostToDevice));
            linear_f32(0, (const float*)dW.p, N, K, (const float*)dX.p, (float*)dY.p, M);
            CK(hipDeviceSynchronize());
            std::vector<float> Y((size_t)M * N);
            CK(hipMemcpy(Y.data(), dY.p, Y.size() * 4, hipMemcpyDeviceToHost));
            double maxv = 0, maxe = 0;
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    double a = 0;
                    for (int k = 0; k < K; ++k) a += (double)W[(size_t)n * K + k] * (double)X[(size_t)m * K + k];
                    maxv = std::max(maxv, std::fabs(a));
                    maxe = std::max(maxe, std::fabs(a - Y[(size_t)m * N + n]));
                }
            const bool ok = maxe / maxv < 1e-5;
            printf("  %-52s N=%5d K=%5d M=%3d  rel_err=%.3g  %s\n", "random F32", N, K, M, maxe / maxv, ok ? "ok" : "FAIL");
            if (ok) ++g_pass; else ++g_fail;
        }
    }
}

struct Trunk {
    strata::artifact::StrataFileReader rd;
    hy4::JVal man;
    std::vector<const hy4::JVal*> tens;
    uint64_t sec1 = 0, sec4 = 0;
    const hy4::JVal* find(const std::string& name) const {
        for (auto* t : tens) if (t->at("name").str() == name) return t;
        return nullptr;
    }
};

static void test_real(Trunk& T) {
    g_rng = 0x5151515151ull;
    printf("== real SQ-64 trunk tensors\n");
    const char* names[] = {
        "model.layers.5.self_attn.q_b_proj.weight",   "model.layers.2.self_attn.kv_b_proj.weight", "model.layers.2.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.routed_expert_down_proj.weight", "model.layers.3.mlp.routed_expert_up_proj.weight",
        "model.layers.5.self_attn.compressed_indexer.key_gate_projection.weight",
        "model.layers.5.self_attn.compressed_indexer.head_weight_projection.weight",
        "model.layers.5.self_attn.compressed_indexer.query_projection.weight",
        "model.dfly_layers.0.self_attn.q_a_proj.weight", "model.layers.1.engram_layer.key_projs.0.weight", "lm_head.weight"};
    for (const char* nm : names) {
        const hy4::JVal* t = T.find(nm);
        if (!t) { printf("  missing tensor %s\n", nm); ++g_fail; continue; }
        const std::string fmts = t->at("format").str();
        int fmt = fmts == "S2_K" ? FMT_S2 : fmts == "S3_K" ? FMT_S3 : fmts == "S4_K" ? FMT_S4 : fmts == "S5_K" ? FMT_S5 : fmts == "S6_K" ? FMT_S6
                : fmts == "S8_K" ? FMT_S8 : fmts == "Q8_ROW" ? FMT_Q8ROW : -1;
        int N = (int)t->at("shape")[0].i(), K = (int)t->at("shape")[1].i();
        const uint64_t off = (uint64_t)t->at("offset").i();
        const size_t bytes = (size_t)t->at("bytes").i();
        std::string err;
        if (fmt == FMT_Q8ROW) {
            std::vector<uint8_t> raw(bytes);
            if (!T.rd.read_at(T.sec1 + off, raw.data(), bytes, err)) { printf("  read fail %s\n", err.c_str()); ++g_fail; continue; }
            test_linear(nm, fmt, raw.data(), bytes, N, K, {1, 3, 8, 16, 100});
            continue;
        }
        // cap rows so that the CPU reference stays small (rows are independent, row r starts at r * nsb * sb)
        const size_t rowb = (size_t)(K / 256) * sk_sb_bytes(fmt);
        int Nt = N;
        while ((size_t)Nt * K > 40u * 1000 * 1000 && Nt > 8) Nt /= 2;
        std::vector<uint8_t> raw((size_t)Nt * rowb);
        if (!T.rd.read_at(T.sec1 + off, raw.data(), raw.size(), err)) { printf("  read fail %s\n", err.c_str()); ++g_fail; continue; }
        test_linear(nm, fmt, raw.data(), raw.size(), Nt, K, {1, 3, 8, 16, 100});
    }
}

struct ExpertTest {
    int layer, rank;
};

static void test_experts(Trunk& T) {
    g_rng = 0x6262626262ull;
    printf("== real SQ-64 expert bodies (section 4)\n");
    const auto& dir = T.rd.expert_directory();
    printf("  directory entries %zu\n", dir.size());
    int counts[16] = {0};
    for (auto& e : dir) { counts[e.gu_code_id & 15]++; }
    printf("  gu code histogram:");
    for (int i = 0; i < 16; ++i) if (counts[i]) printf(" S%d=%d", i, counts[i]);
    printf("\n");
    // one expert of every distinct (gu,dn) code pair that appears
    std::vector<std::pair<int, int>> seen;
    std::vector<size_t> lasti;
    for (size_t i = 0; i < dir.size(); i += 1) {
        const auto& e = dir[i];
        std::pair<int, int> key(e.gu_code_id, e.dn_code_id);
        const long cnt = (long)std::count(seen.begin(), seen.end(), key);
        if (cnt >= 2) continue;
        if (cnt == 1) {
            size_t li = 0;
            for (size_t j = 0; j < seen.size(); ++j) if (seen[j] == key) li = lasti[j];
            if (i < li + 5000) continue;
        }
        seen.push_back(key);
        lasti.push_back(i);
        std::vector<uint8_t> body;
        std::string err;
        if (!T.rd.read_expert_body(i, body, err)) { printf("  read expert %zu failed: %s\n", i, err.c_str()); ++g_fail; continue; }
        const int guf = e.gu_code_id, dnf = e.dn_code_id;
        char nm[96];
        snprintf(nm, sizeof nm, "expert #%zu (L%zu rank %zu) gate_up S%d", i, i / 512, i % 512, guf);
        test_linear(nm, guf, body.data() + 3072, weight_bytes(guf, 3072, 1536), 3072, 1536, {1, 3, 8, 16, 100});
        snprintf(nm, sizeof nm, "expert #%zu (L%zu rank %zu) down S%d", i, i / 512, i % 512, dnf);
        test_linear(nm, dnf, body.data() + e.dn_offset, weight_bytes(dnf, 1536, 1536), 1536, 1536, {1, 3, 8, 16, 100});
    }
}

static void test_embed(Trunk& T) {
    printf("== embed_rows\n");
    const hy4::JVal* t = T.find("model.embed_tokens.weight");
    if (!t) { ++g_fail; return; }
    const int N = (int)t->at("shape")[0].i(), K = (int)t->at("shape")[1].i();
    const size_t bytes = (size_t)t->at("bytes").i();
    std::vector<uint8_t> raw(bytes);
    std::string err;
    if (!T.rd.read_at(T.sec1 + (uint64_t)t->at("offset").i(), raw.data(), bytes, err)) { printf("  read fail\n"); ++g_fail; return; }
    std::vector<int32_t> ids = {0, 1, 2, 120000, 120001, 120831, 5, 99999, 4096, 77};
    std::vector<float> cpu(ids.size() * (size_t)K);
    const size_t rb = (size_t)(K / 256) * sk_sb_bytes(FMT_S8);
    for (size_t i = 0; i < ids.size(); ++i)
        strata::quant::sk::dequant_sk_row(raw.data() + (size_t)ids[i] * rb, strata::quant::sk::SkCode::S8_K, cpu.data() + i * K, K);
    Dev dW(bytes), dI(ids.size() * 4), dO(cpu.size() * 4);
    CK(hipMemcpy(dW.p, raw.data(), bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dI.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
    embed_rows(0, FMT_S8, dW.p, N, K, (const int32_t*)dI.p, (float*)dO.p, (int)ids.size());
    CK(hipDeviceSynchronize());
    std::vector<float> got(cpu.size());
    CK(hipMemcpy(got.data(), dO.p, got.size() * 4, hipMemcpyDeviceToHost));
    double maxe = 0;
    for (size_t i = 0; i < got.size(); ++i) maxe = std::max(maxe, (double)std::fabs(got[i] - cpu[i]));
    printf("  embed S8 rows: max abs err %.3g  %s\n", maxe, maxe == 0 ? "ok (exact)" : (maxe < 1e-6 ? "ok" : "FAIL"));
    if (maxe < 1e-6) ++g_pass; else ++g_fail;
}

int main(int argc, char** argv) {
    std::string path = "C:\\Hy4-Flash\\Strata-Quant-Hy4Flash-pack\\store\\Hy4-Flash-SQ-64.strata";
    std::string mode = "all";
    bool random_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--strata" && i + 1 < argc) path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--random-only") random_only = true;
    }
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("device %s %s warp %d CUs %d\n", p.name, gpu_arch(p), p.warpSize, p.multiProcessorCount);
    test_random_codes();
    if (random_only) {
        printf("\nSUMMARY pass=%d fail=%d\n", g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
    Trunk T;
    std::string err;
    if (!T.rd.open(path, err)) { printf("open failed: %s\n", err.c_str()); return 1; }
    T.man = hy4::json_parse(T.rd.manifest_json());
    for (auto& t : T.man.at("trunk_tensors").a) T.tens.push_back(&t);
    T.sec1 = T.rd.header().sections[1].file_offset;
    T.sec4 = T.rd.header().sections[4].file_offset;
    test_real(T);
    test_experts(T);
    test_embed(T);
    printf("\nSUMMARY pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
