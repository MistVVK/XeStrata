// src/kernels/gr_v3_screen.cpp - the v3 hyper-connection read (gr_down_v3 / gr_up_v3, the sm_120 default) with and
// without its weight-prefetch variants (STRATA_GR_V3_PF; BW-DENSE): every output of every variant must be bitwise the
// plain v3 read's (FNV-1a hash of R, lo, rs, inject and mixed), and each variant's time on COLD weights (the read
// rotates over `sets` copies of one layer's 13 MB so L2 holds none of it, as in the engine).
//
//     build/gr_v3_screen [iters] [T min] [T max] [sets] [rounds]
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)   // MSVC has no setenv / unsetenv
#define setenv(k, v, o) _putenv_s((k), (v))
#define unsetenv(k) _putenv_s((k), "")
#endif

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setenv("STRATA_GR_V3", "1", 1);
    setvbuf(stdout, nullptr, _IONBF, 0);
    K::fused_gr_v3_print_resources();
    if (argc > 1 && std::string(argv[1]) == "--resources") return 0;
    const int iters = argc > 1 ? std::atoi(argv[1]) : 400;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 8;
    const int sets = argc > 4 ? std::atoi(argv[4]) : 16;
    const int rounds = argc > 5 ? std::atoi(argv[5]) : 5;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dxn, *dlo, *drs, *dio, *dmix;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4 * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    std::vector<uint16_t*> cwd(sets), cwu(sets), cwi(sets);
    std::vector<float*> cwn(sets);
    for (int c = 0; c < sets; ++c) {
        cudaMalloc((void**) &cwd[c], wd.size() * 2); cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &cwu[c], wu.size() * 2); cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &cwi[c], wi.size() * 2); cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &cwn[c], wn.size() * 4); cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    unsigned long long* dst;
    cudaMalloc((void**) &dst, (size_t) iters * 4 * 8);
    std::vector<unsigned long long> hst((size_t) iters * 4);
    std::vector<K::FusedGrArgs> as((size_t) sets * 8);
    const int masks[] = {0, 5, 133, 261};
    const int NM = (int) (sizeof(masks) / sizeof(masks[0]));
    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T) {
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                for (int c = 0; c < sets; ++c)
                    for (int t = 0; t < T; ++t) {
                        K::FusedGrArgs& a = as[(size_t) c * 8 + t];
                        a = K::FusedGrArgs();
                        a.R = dR + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = apply;
                        a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                        a.w_norm = cwn[c]; a.w_down = cwd[c]; a.w_up = cwu[c]; a.w_inject = inject ? cwi[c] : nullptr;
                        a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                        a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
                    }
                unsigned long long h0 = 0;
                for (int mi = 0; mi < NM; ++mi) {   // bit for bit against mask 0, on the last weight set as well as the first
                    unsigned long long h = 1469598103934665603ull;
                    for (int cset : {0, sets - 1}) {
                        K::fused_gr_v3_pf(masks[mi]);
                        cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                        cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                        cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                        cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                        cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                        K::fused_gr_read_multi(&as[(size_t) cset * 8], T, dxn, s);
                        cudaStreamSynchronize(s);
                        std::vector<float> o((size_t) TM * (D + LR + HC + HC + N));
                        float* p = o.data();
                        cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                        cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                        cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                        for (size_t b = 0; b < o.size() * 4; ++b) h = (h ^ ((const unsigned char*) o.data())[b]) * 1099511628211ull;
                    }
                    if (mi == 0) h0 = h;
                    else if (h != h0) {
                        ++failures;
                        std::printf("FAIL T %d apply %d inject %d: mask %d differs from the plain v3 read\n", T, apply, inject, masks[mi]);
                    }
                }
            }
        // timing: apply = 1, inject = 1 (a layer read), cold weights; the variants interleaved, the median of the rounds
        for (int c = 0; c < sets; ++c)
            for (int t = 0; t < T; ++t) {
                K::FusedGrArgs& a = as[(size_t) c * 8 + t];
                a = K::FusedGrArgs();
                a.R = dR + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = 1;
                a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                a.w_norm = cwn[c]; a.w_down = cwd[c]; a.w_up = cwu[c]; a.w_inject = cwi[c];
                a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
            }
        std::vector<double> tot[16], dn[16], up[16];
        for (int r = 0; r < rounds; ++r)
            for (int mi = 0; mi < NM; ++mi) {
                K::fused_gr_v3_pf(masks[mi]);
                for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % sets) * 8], T, dxn, s);   // warm
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % sets) * 8], T, dxn, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                tot[mi].push_back(1e3 * ms / iters);
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi(&as[(size_t) (i % sets) * 8], T, dxn, s, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                cudaStreamSynchronize(s);
                cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost);
                std::vector<double> d_, u_;
                for (int i = 1; i < iters; ++i) {
                    const unsigned long long* p = &hst[(size_t) i * 4];
                    d_.push_back((double) (p[2] - p[1]));
                    u_.push_back((double) (p[3] - p[2]));
                }
                std::sort(d_.begin(), d_.end());
                std::sort(u_.begin(), u_.end());
                dn[mi].push_back(d_[d_.size() / 2] / 1e3);
                up[mi].push_back(u_[u_.size() / 2] / 1e3);
            }
        auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::printf("T %d cold(%d sets) read us:", T, sets);
        for (int mi = 0; mi < NM; ++mi) std::printf("  pf%d %6.2f (down %5.2f up %5.2f)", masks[mi], med(tot[mi]), med(dn[mi]), med(up[mi]));
        std::printf("   [stamps include a ~1-2 us stamp kernel each]\n");
    }
    K::fused_gr_v3_pf(-1);
    std::printf("gr_v3_screen: %d differing\n", failures);
    return failures ? 1 : 0;
}
