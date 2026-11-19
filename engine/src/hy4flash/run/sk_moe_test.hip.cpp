// sk_moe_test.hip.cpp - moe_experts() vs a CPU double reference
//   (1) real SQ-64 expert bodies of one layer (S2/S3/S4 codes)
//   (2) synthetic bodies in the real body layout (sq_xd_dn fp16[1536] | gate_up [3072x1536] code gu | down [1536x1536] code dn | 4 KiB pad) for every
//       code pair the four SQ builds use: S1.25+S1.25, S1.5+S1.5, S2+S2, S2+S3, S3+S3, S3+S4, S4+S4, S4+S5, S5+S6, S6+S8, S8+S8 (mixed in one layer)
// Prints an FNV hash of z_out per case so two builds can be compared bit for bit (-DSK_OLD builds skip the new codes).
#include "skgemm.hip.h"
#include "skmoe.hip.h"
#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace hy4;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint64_t g_rng = 0x1234567ull;
static uint32_t rnd() { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static float rndf() { return (rnd() & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f; }

static uint64_t fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Ent { int gu, dn; uint32_t dn_off, body_bytes; };

// one case: M tokens, 16 distinct experts per token among the first `pool` expert ids
static int run_case(const char* label, int M, int pool, const std::vector<Ent>& ents, const std::function<void(int, std::vector<uint8_t>&)>& get_body,
                    int dec_max) {
    const int P = M * 16;
    std::vector<int32_t> idx((size_t)P);
    std::vector<float> wts((size_t)P);
    std::set<int> used;
    for (int t = 0; t < M; ++t) {
        std::set<int> s;
        const int lim = (t % 3 == 0 && pool > 64) ? 64 : pool;
        while ((int)s.size() < 16) s.insert((int)(rnd() % lim));
        int k = 0;
        for (int e : s) { idx[(size_t)t * 16 + k] = e; wts[(size_t)t * 16 + k] = 0.05f + (rnd() % 1000) / 4000.0f; used.insert(e); ++k; }
    }
    std::vector<ExpertDev> tab(512);
    std::memset(tab.data(), 0, sizeof(ExpertDev) * 512);
    size_t arena_bytes = 0;
    std::vector<std::vector<uint8_t>> bodies(512);
    for (int e : used) {
        get_body(e, bodies[e]);
        tab[e].body_off = arena_bytes;
        tab[e].dn_off = ents[e].dn_off;
        tab[e].gu_code = (uint8_t)ents[e].gu;
        tab[e].dn_code = (uint8_t)ents[e].dn;
        arena_bytes += (bodies[e].size() + 4095) & ~(size_t)4095;
    }
    uint8_t* d_arena;
    CK(hipMalloc(&d_arena, arena_bytes + 4096));
    for (int e : used) CK(hipMemcpy(d_arena + tab[e].body_off, bodies[e].data(), bodies[e].size(), hipMemcpyHostToDevice));
    ExpertDev* d_tab;
    CK(hipMalloc(&d_tab, sizeof(ExpertDev) * 512));
    CK(hipMemcpy(d_tab, tab.data(), sizeof(ExpertDev) * 512, hipMemcpyHostToDevice));
    std::vector<__half> z((size_t)M * 1536);
    for (auto& v : z) v = __float2half(rndf() * 1.5f);
    __half* d_z; int32_t* d_idx; float *d_w, *d_out; void* d_ws;
    CK(hipMalloc(&d_z, z.size() * 2));
    CK(hipMalloc(&d_idx, idx.size() * 4));
    CK(hipMalloc(&d_w, wts.size() * 4));
    CK(hipMalloc(&d_out, (size_t)M * 1536 * 4));
    CK(hipMalloc(&d_ws, moe_ws_bytes(M)));
    CK(hipMemcpy(d_z, z.data(), z.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_idx, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_w, wts.data(), wts.size() * 4, hipMemcpyHostToDevice));
#ifndef SK_OLD
    skd::moe_dec_max_m() = dec_max;
#else
    (void)dec_max;
#endif
    hipEvent_t ea, eb;
    hipEventCreate(&ea); hipEventCreate(&eb);
    moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
    CK(hipDeviceSynchronize());
    hipEventRecord(ea);
    const int reps = 5;
    for (int r = 0; r < reps; ++r) moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
    hipEventRecord(eb);
    CK(hipEventSynchronize(eb));
    float ms = 0; hipEventElapsedTime(&ms, ea, eb);
    std::vector<float> got((size_t)M * 1536), got2((size_t)M * 1536);
    CK(hipMemcpy(got.data(), d_out, got.size() * 4, hipMemcpyDeviceToHost));
    moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(got2.data(), d_out, got2.size() * 4, hipMemcpyDeviceToHost));
    const bool det = std::memcmp(got.data(), got2.data(), got.size() * 4) == 0;

    std::vector<double> ref((size_t)M * 1536, 0.0);
    for (int e : used) {
        std::vector<float> Wgu((size_t)3072 * 1536), Wdn((size_t)1536 * 1536);
        const size_t rbg = 6 * sk_sb_bytes(ents[e].gu), rbd = 6 * sk_sb_bytes(ents[e].dn);
        for (int r = 0; r < 3072; ++r)
            strata::quant::sk::dequant_sk_row(bodies[e].data() + 3072 + r * rbg, (strata::quant::sk::SkCode)ents[e].gu, &Wgu[(size_t)r * 1536], 1536);
        for (int r = 0; r < 1536; ++r)
            strata::quant::sk::dequant_sk_row(bodies[e].data() + ents[e].dn_off + r * rbd, (strata::quant::sk::SkCode)ents[e].dn, &Wdn[(size_t)r * 1536], 1536);
        std::vector<float> sqx(1536);
        for (int i = 0; i < 1536; ++i) sqx[i] = __half2float(((const __half*)bodies[e].data())[i]);
        for (int t = 0; t < M; ++t)
            for (int s = 0; s < 16; ++s) {
                if (idx[(size_t)t * 16 + s] != e) continue;
                std::vector<double> h(1536);
                for (int r = 0; r < 1536; ++r) {
                    double g = 0, u = 0;
                    for (int k = 0; k < 1536; ++k) {
                        const double x = __half2float(z[(size_t)t * 1536 + k]);
                        g += Wgu[(size_t)r * 1536 + k] * x;
                        u += Wgu[(size_t)(1536 + r) * 1536 + k] * x;
                    }
                    const double sg = g / (1.0 + std::exp(-g));
                    h[r] = std::min(sg, 10.0) * std::min(std::max(u, -10.0), 10.0) / sqx[r];
                }
                const double w = wts[(size_t)t * 16 + s];
                for (int n = 0; n < 1536; ++n) {
                    double a = 0;
                    for (int k = 0; k < 1536; ++k) a += Wdn[(size_t)n * 1536 + k] * h[k];
                    ref[(size_t)t * 1536 + n] += w * a;
                }
            }
    }
    double maxv = 0, maxe = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxv = std::max(maxv, std::fabs(ref[i]));
        double d = std::fabs(ref[i] - got[i]);
        if (!(d == d)) d = 1e30;
        maxe = std::max(maxe, d);
    }
    const double rel = maxv > 0 ? maxe / maxv : maxe;
    const bool ok = rel < 2e-3 && det;
    printf("%-9s dec_max=%2d M=%3d pairs=%4d experts=%3zu max|z_out|=%.4g rel_err=%.3g det=%d %.3f ms/call hash=%016llx %s\n", label, dec_max, M, P, used.size(), maxv,
           rel, det ? 1 : 0, ms / reps, (unsigned long long)fnv(got.data(), got.size() * 4), ok ? "ok" : "FAIL");
    hipFree(d_arena); hipFree(d_tab); hipFree(d_z); hipFree(d_idx); hipFree(d_w); hipFree(d_out); hipFree(d_ws);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string path = "C:\\Hy4-Flash\\Strata-Quant-Hy4Flash-pack\\store\\Hy4-Flash-SQ-64.strata";
    int layer = 3;
    bool real = true, synth = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--strata" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = atoi(argv[++i]);
        else if (a == "--real-only") synth = false;
        else if (a == "--synth-only") real = false;
    }
    int fails = 0;
    if (real) {
        strata::artifact::StrataFileReader rd;
        std::string err;
        if (!rd.open(path, err)) { printf("open failed: %s\n", err.c_str()); return 1; }
        const auto& dir = rd.expert_directory();
        const uint64_t sec4 = rd.header().sections[4].file_offset;
        std::vector<Ent> ents(512);
        std::vector<const strata::artifact::StrataExpertEntry*> by_id(512, nullptr);
        for (int rk = 0; rk < 512; ++rk) {
            const auto& e = dir[(size_t)layer * 512 + rk];
            by_id[e.expert_id] = &e;
            ents[e.expert_id] = {e.gu_code_id, e.dn_code_id, e.dn_offset, e.body_bytes};
        }
        auto get = [&](int e, std::vector<uint8_t>& b) {
            b.resize(by_id[e]->body_bytes);
            std::string er;
            if (!rd.read_at(sec4 + by_id[e]->body_offset, b.data(), b.size(), er)) { printf("read failed %s\n", er.c_str()); exit(1); }
        };
        g_rng = 0x1234567ull;
        for (int M : {1, 2, 3, 8, 9, 32, 130}) fails += run_case("real", M, 512, ents, get, 4);
#ifndef SK_OLD
        // decode kernels for the verify windows (M <= 16), the option that is off by default
        g_rng = 0x7777ull;
        for (int M : {5, 8, 12, 16}) fails += run_case("real-dec", M, 512, ents, get, 16);
#endif
    }
#ifndef SK_OLD
    if (synth) {
        const int pairs[][2] = {{0, 0}, {1, 1}, {2, 2}, {2, 3}, {3, 3}, {3, 4}, {4, 4}, {4, 5}, {5, 6}, {6, 8}, {8, 8}, {0, 0}};
        const int npairs = sizeof(pairs) / sizeof(pairs[0]);
        const int pool = 48;
        std::vector<Ent> ents(512);
        for (int e = 0; e < pool; ++e) {
            const int gu = pairs[e % npairs][0], dn = pairs[e % npairs][1];
            const uint32_t dn_off = 3072 + 18432u * sk_sb_bytes(gu);
            const uint32_t bb = (uint32_t)((dn_off + 9216ull * sk_sb_bytes(dn) + 4095) & ~4095ull);
            ents[e] = {gu, dn, dn_off, bb};
        }
        auto fill_sb = [](uint8_t* blk, int sb, uint64_t& st) {
            for (int i = 0; i < sb; ++i) { st = st * 6364136223846793005ull + 1442695040888963407ull; blk[i] = (uint8_t)(st >> 40); }
            const __half h = __float2half(0.002f + (float)(blk[7] & 255) / 255.0f * 0.03f);
            std::memcpy(blk, &h, 2);
            for (int g = 0; g < 4; ++g) blk[2 + g] = (uint8_t)(64 + (blk[2 + g] & 191));
        };
        auto get = [&](int e, std::vector<uint8_t>& b) {
            b.assign(ents[e].body_bytes, 0);
            uint64_t st = 0xC0FFEEull + 977ull * e;
            for (int i = 0; i < 1536; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                const __half h = __float2half(0.5f + (float)((st >> 40) & 1023) / 1024.0f);
                std::memcpy(b.data() + 2 * i, &h, 2);
            }
            const int sg = sk_sb_bytes(ents[e].gu), sd = sk_sb_bytes(ents[e].dn);
            for (int k = 0; k < 18432; ++k) fill_sb(b.data() + 3072 + (size_t)k * sg, sg, st);
            for (int k = 0; k < 9216; ++k) fill_sb(b.data() + ents[e].dn_off + (size_t)k * sd, sd, st);
        };
        g_rng = 0xABCDEFull;
        for (int M : {1, 3, 8, 16, 40}) fails += run_case("synth", M, pool, ents, get, 4);
        g_rng = 0xABCDEFull;
        for (int M : {3, 8, 16}) fails += run_case("synth-dec", M, pool, ents, get, 16);
    }
#endif
    printf("MOE SUMMARY fails=%d\n", fails);
    return fails ? 1 : 0;
}
