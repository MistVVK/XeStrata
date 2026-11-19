// sk_moe_test.hip.cpp - moe_experts() vs a CPU double reference on real SQ-64 expert bodies
#include "skgemm.hip.h"
#include "skmoe.hip.h"
#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace hy4;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint64_t g_rng = 0x1234567ull;
static uint32_t rnd() { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static float rndf() { return (rnd() & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f; }

int main(int argc, char** argv) {
    std::string path = "C:\\Hy4-Flash\\Strata-Quant-Hy4Flash-pack\\store\\Hy4-Flash-SQ-64.strata";
    int layer = 3;
    std::vector<int> Ms = {1, 2, 3, 8, 9, 32, 130};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--strata" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = atoi(argv[++i]);
    }
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(path, err)) { printf("open failed: %s\n", err.c_str()); return 1; }
    const auto& dir = rd.expert_directory();
    const uint64_t sec4 = rd.header().sections[4].file_offset;
    // per expert id of this MoE layer: directory entry (entries are in rank order)
    std::vector<const strata::artifact::StrataExpertEntry*> by_id(512, nullptr);
    for (int rk = 0; rk < 512; ++rk) {
        const auto& e = dir[(size_t)layer * 512 + rk];
        by_id[e.expert_id] = &e;
    }
    int fails = 0;
    for (int M : Ms) {
        const int P = M * 16;
        // routing: 16 distinct experts per token, skewed so that some experts repeat across tokens
        std::vector<int32_t> idx((size_t)P);
        std::vector<float> wts((size_t)P);
        std::set<int> used;
        for (int t = 0; t < M; ++t) {
            std::set<int> s;
            while ((int)s.size() < 16) s.insert((int)(rnd() % (t % 3 == 0 ? 64 : 512)));
            int k = 0;
            for (int e : s) { idx[(size_t)t * 16 + k] = e; wts[(size_t)t * 16 + k] = 0.05f + (rnd() % 1000) / 4000.0f; used.insert(e); ++k; }
        }
        // arena with the used expert bodies
        std::vector<ExpertDev> tab(512);
        std::memset(tab.data(), 0, sizeof(ExpertDev) * 512);
        size_t arena_bytes = 0;
        std::vector<std::vector<uint8_t>> bodies(512);
        for (int e : used) {
            const auto* en = by_id[e];
            bodies[e].resize(en->body_bytes);
            if (!rd.read_at(sec4 + en->body_offset, bodies[e].data(), en->body_bytes, err)) { printf("read failed %s\n", err.c_str()); return 1; }
            tab[e].body_off = arena_bytes;
            tab[e].dn_off = en->dn_offset;
            tab[e].gu_code = en->gu_code_id;
            tab[e].dn_code = en->dn_code_id;
            arena_bytes += (en->body_bytes + 4095) & ~(size_t)4095;
        }
        uint8_t* d_arena;
        CK(hipMalloc(&d_arena, arena_bytes));
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
        // reference path: timing and difference to the fast path
        moe_experts_ref(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
        CK(hipDeviceSynchronize());
        hipEventRecord(ea);
        for (int r = 0; r < 3; ++r) moe_experts_ref(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
        hipEventRecord(eb);
        CK(hipEventSynchronize(eb));
        float ms_ref = 0; hipEventElapsedTime(&ms_ref, ea, eb);
        std::vector<float> gref((size_t)M * 1536);
        CK(hipMemcpy(gref.data(), d_out, gref.size() * 4, hipMemcpyDeviceToHost));
        double dref = 0, mref = 0;
        for (size_t i = 0; i < gref.size(); ++i) { dref = std::max(dref, (double)std::fabs(gref[i] - got[i])); mref = std::max(mref, (double)std::fabs(gref[i])); }
        printf("  ref path: %.3f ms/call, max diff fast-vs-ref %.3g (rel %.3g)\n", ms_ref / 3, dref, dref / mref);

        // CPU double reference
        std::vector<double> ref((size_t)M * 1536, 0.0);
        for (int e : used) {
            const auto* en = by_id[e];
            std::vector<float> Wgu((size_t)3072 * 1536), Wdn((size_t)1536 * 1536);
            const size_t rbg = 6 * sk_sb_bytes(en->gu_code_id), rbd = 6 * sk_sb_bytes(en->dn_code_id);
            for (int r = 0; r < 3072; ++r)
                strata::quant::sk::dequant_sk_row(bodies[e].data() + 3072 + r * rbg, (strata::quant::sk::SkCode)en->gu_code_id, &Wgu[(size_t)r * 1536], 1536);
            for (int r = 0; r < 1536; ++r)
                strata::quant::sk::dequant_sk_row(bodies[e].data() + en->dn_offset + r * rbd, (strata::quant::sk::SkCode)en->dn_code_id, &Wdn[(size_t)r * 1536], 1536);
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
        for (size_t i = 0; i < ref.size(); ++i) { maxv = std::max(maxv, std::fabs(ref[i])); maxe = std::max(maxe, std::fabs(ref[i] - got[i])); }
        const double rel = maxe / maxv;
        const bool ok = rel < 2e-3 && det;
        printf("moe_experts M=%3d pairs=%4d experts_used=%3zu  max|z_out|=%.4g rel_err=%.3g deterministic=%d  time/call=%.3f ms  %s\n", M, P, used.size(), maxv, rel,
               det ? 1 : 0, ms / reps, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
        hipFree(d_arena); hipFree(d_tab); hipFree(d_z); hipFree(d_idx); hipFree(d_w); hipFree(d_out); hipFree(d_ws);
    }
    printf("MOE SUMMARY fails=%d\n", fails);
    return fails ? 1 : 0;
}
