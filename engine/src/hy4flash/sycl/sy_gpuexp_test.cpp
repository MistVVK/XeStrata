// sy_gpuexp_test.cpp - GpuExperts vs CpuExperts::run (act_mode 0) on the real SQ-32 file, plus timing.
// build: HY4S_LIBS=1 b_sycl.sh <tag> sy_gpuexp_test sycl/sy_gpuexp_test.cpp sycl/hy4s_cpuexp.cpp sycl/sy_gpuexp.cpp
// run:   sy_gpuexp_test <SQ-32.strata> [stage_MiB=384] [threads=6]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "sy_gpuexp.h"

using namespace hy4s;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Route {
    std::vector<int32_t> idx;
    std::vector<float> w;
};

static Route make_route(const CpuExperts& cx, int L, int T, bool skewed, std::mt19937& rng) {
    // rank of every expert of the layer = order of the body offsets (directory order is rank order)
    std::vector<std::pair<const uint8_t*, int>> bo;
    for (int e = 0; e < 512; ++e) {
        const uint8_t* b; int gu, dn; uint32_t d, n;
        cx.expert_info(L, e, &b, &gu, &dn, &d, &n);
        bo.push_back({b, e});
    }
    std::sort(bo.begin(), bo.end());
    std::vector<int> by_rank(512);
    for (int r = 0; r < 512; ++r) by_rank[r] = bo[r].second;
    Route R;
    R.idx.resize((size_t)T * 16);
    R.w.resize((size_t)T * 16);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick;
        while ((int)pick.size() < 16) {
            int e;
            if (skewed) {  // P(rank r) ~ 1/(1 + r/24): 45% of the hits on the first 80 ranks
                const double u = std::uniform_real_distribution<double>(0, 1)(rng);
                const double H = std::log(1.0 + 511.0 / 24.0);
                const int r = std::min(511, (int)(24.0 * (std::exp(u * H) - 1.0)));
                e = by_rank[r];
            } else {
                e = (int)(rng() % 512);
            }
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        float s = 0;
        for (int k = 0; k < 16; ++k) {
            R.idx[(size_t)t * 16 + k] = pick[k];
            R.w[(size_t)t * 16 + k] = std::uniform_real_distribution<float>(0.2f, 1.0f)(rng);
            s += R.w[(size_t)t * 16 + k];
        }
        for (int k = 0; k < 16; ++k) R.w[(size_t)t * 16 + k] = R.w[(size_t)t * 16 + k] / s * 3.998f;
    }
    return R;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.strata [stage_MiB] [threads]\n", argv[0]);
        return 2;
    }
    const size_t stage = (size_t)(argc > 2 ? atoi(argv[2]) : 384) << 20;
    const int threads = argc > 3 ? atoi(argv[3]) : 6;
    std::string err;
    CpuExperts cx;
    if (!cx.open(argv[1], threads, err, 512)) {
        fprintf(stderr, "cx.open: %s\n", err.c_str());
        return 1;
    }
    double ws = 0;
    cx.warm(threads, &ws);
    GpuExperts gx;
    if (!gx.init(cx, stage, 512, err, threads)) {
        fprintf(stderr, "gx.init: %s\n", err.c_str());
        return 1;
    }
    printf("device %s; GpuExperts device allocation %.1f MiB (ring 2 x %.0f MiB)\n", Q().get_device().get_info<sycl::info::device::name>().c_str(), gx.device_bytes() / 1048576.0,
           (stage / 3 & ~(size_t)4095) / 1048576.0);
    std::mt19937 rng(2026);
    int fails = 0;
    const int maxT = 512;
    float* d_z = dalloc<float>((size_t)maxT * 1536);
    float* d_out = dalloc<float>((size_t)maxT * 1536);
    std::vector<float> z((size_t)maxT * 1536), ref((size_t)maxT * 1536), got((size_t)maxT * 1536);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (auto& v : z) v = nd(rng);
    h2d(d_z, z.data(), z.size() * 4);
    wait();
    // ---------------- accuracy
    for (int L : {0, 7, 20, 40})
        for (int T : {64, 512})
            for (int skew = 0; skew < 2; ++skew) {
                Route R = make_route(cx, L, T, skew != 0, rng);
                cx.run(L, T, z.data(), R.idx.data(), R.w.data(), ref.data(), 0);
                gx.run(L, T, d_z, R.idx.data(), R.w.data(), d_out, nullptr);
                d2h(got.data(), d_out, (size_t)T * 1536 * 4);
                gx.wait_all();
                double num = 0, den = 0, mx = 0, rmx = 0;
                for (size_t i = 0; i < (size_t)T * 1536; ++i) {
                    const double d = (double)got[i] - ref[i];
                    num += d * d;
                    den += (double)ref[i] * ref[i];
                    mx = std::max(mx, std::fabs(d));
                    rmx = std::max(rmx, (double)std::fabs(ref[i]));
                }
                const double rel = std::sqrt(num / den);
                printf("layer %2d T=%3d %-7s rel L2 %.3e  max abs err %.3e (max |ref| %.3e)  %s\n", L, T, skew ? "skewed" : "uniform", rel, mx, rmx, rel <= 1e-3 ? "ok" : "FAIL");
                if (rel > 1e-3) ++fails;
            }
    {  // only mask: even experts only; the CPU reference gets weight 0 for the masked pairs
        const int L = 20, T = 512;
        Route R = make_route(cx, L, T, false, rng);
        std::vector<uint8_t> only(512);
        for (int e = 0; e < 512; ++e) only[e] = (e % 2 == 0);
        Route R2 = R;
        for (size_t i = 0; i < R.idx.size(); ++i)
            if (!only[R.idx[i]]) R2.w[i] = 0.0f;
        cx.run(L, T, z.data(), R2.idx.data(), R2.w.data(), ref.data(), 0);
        gx.run(L, T, d_z, R.idx.data(), R.w.data(), d_out, only.data());
        d2h(got.data(), d_out, (size_t)T * 1536 * 4);
        gx.wait_all();
        double num = 0, den = 0;
        for (size_t i = 0; i < (size_t)T * 1536; ++i) {
            const double d = (double)got[i] - ref[i];
            num += d * d;
            den += (double)ref[i] * ref[i];
        }
        const double rel = std::sqrt(num / den);
        printf("layer %2d T=%3d only-even-experts mask rel L2 %.3e  %s\n", L, T, rel, rel <= 1e-3 ? "ok" : "FAIL");
        if (rel > 1e-3) ++fails;
    }
    // ---------------- timing: all 41 layers at T=512, uniform and skewed (stats via wall time with a wait at the end)
    for (int skew = 0; skew < 2; ++skew) {
        std::vector<Route> routes;
        for (int L = 0; L < 41; ++L) routes.push_back(make_route(cx, L, 512, skew != 0, rng));
        for (int pass = 0; pass < 2; ++pass) {
            gx.reset_stats();
            wait();
            const double t0 = now();
            for (int L = 0; L < 41; ++L) gx.run(L, 512, d_z, routes[L].idx.data(), routes[L].w.data(), d_out, nullptr);
            gx.wait_all();
            const double dt = now() - t0;
            const auto& s = gx.stats();
            printf("%-7s pass %d: 41 layers T=512 in %.3f s (%.1f ms/layer; CPU path ~9.3 s): bytes %.2f GB, host copy %.3f s (%.2f GB/s), plan %.3f s (of which waiting for the previous layer %.3f s), host slot wait %.3f s, groups %llu, experts/layer %.0f, pairs %llu\n",
                   skew ? "skewed" : "uniform", pass, dt, dt * 1e3 / 41, s.bytes / 1e9, s.t_copy_host, s.bytes / 1e9 / std::max(s.t_copy_host, 1e-9), s.t_plan, s.t_wait_prev, s.t_wait_host, (unsigned long long)s.groups,
                   (double)s.experts / 41, (unsigned long long)s.pairs);
        }
    }
    // ---------------- device-time profile with a profiling compute queue (a few layers, uniform + skewed)
    {
        sycl::queue pq(Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        gx.set_profile_queue(&pq);
        for (int skew = 0; skew < 2; ++skew) {
            gx.reset_stats();
            const int Ls[] = {5, 12, 20, 33};
            for (int L : Ls) {
                Route R = make_route(cx, L, 512, skew != 0, rng);
                gx.run(L, 512, d_z, R.idx.data(), R.w.data(), d_out, nullptr);
            }
            gx.wait_all();
            pq.wait();
            const auto& s = gx.stats();
            printf("profile %-7s (4 layers, T=512): GEMM1 %.1f ms, GEMM2 %.1f ms, other kernels+uploads %.2f ms, H2D %.1f ms (%.2f GB -> %.1f GB/s on the device clock) per layer avg: gemm1 %.1f gemm2 %.1f h2d %.1f\n",
                   skew ? "skewed" : "uniform", s.gpu_gemm1_ms, s.gpu_gemm2_ms, s.gpu_other_ms, s.gpu_h2d_ms, s.bytes / 1e9, s.bytes / 1e9 / (s.gpu_h2d_ms * 1e-3 + 1e-12), s.gpu_gemm1_ms / 4, s.gpu_gemm2_ms / 4,
                   s.gpu_h2d_ms / 4);
        }
        gx.set_profile_queue(nullptr);
    }
    printf("%s (%d failures)\n", fails ? "SOME FAILED" : "ALL OK", fails);
    return fails ? 1 : 0;
}
