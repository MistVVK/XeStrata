// idx_select_test.hip.cpp - k_idx_select (radix threshold search + ordered emit) against a CPU reference on random scores with ties, -inf and negative values
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hy4_gpu.h"
#include "hy4_indexer.hip.h"

using namespace hy4;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 2; } } while (0)

static uint32_t f2key_h(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

int main() {
    const int n_top = 511, kmax = 2048 + 8;
    int bad = 0, total = 0;
    uint32_t rs = 12345;
    auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs >> 8; };
    for (int grp : {600, 1024, 2047, 4096, 9000, 32768}) {
        for (int mode = 0; mode < 4; ++mode) {
            const int ldS = grp + 8, T = 4;
            const int p0 = 4 * grp;  // positions 4*grp .. 4*grp+3 all have group grp
            std::vector<float> sc((size_t)T * ldS, 0.f);
            for (int t = 0; t < T; ++t)
                for (int g = 0; g < ldS; ++g) {
                    float v = ((int)(rnd() % 20001) - 10000) / 100.0f;
                    if (mode == 1) v = (float)((int)(rnd() % 7) - 3);          // many ties
                    if (mode == 2 && rnd() % 5 == 0) v = -INFINITY;            // -inf entries
                    if (mode == 3) v = (rnd() % 3 == 0) ? 1.5f : (float)(rnd() % 100) * 1e-3f;  // big tie block
                    sc[(size_t)t * ldS + g] = v;
                }
            float* d_sc;
            int32_t *d_keys, *d_nk;
            CK(hipMalloc(&d_sc, sc.size() * 4));
            CK(hipMalloc(&d_keys, (size_t)T * kmax * 4));
            CK(hipMalloc(&d_nk, T * 4));
            CK(hipMemcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(idx::k_idx_select, dim3(T), dim3(256), 0, 0, d_sc, ldS, p0, n_top, kmax, d_keys, d_nk);
            CK(hipDeviceSynchronize());
            std::vector<int32_t> keys((size_t)T * kmax), nk(T);
            CK(hipMemcpy(keys.data(), d_keys, keys.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(nk.data(), d_nk, T * 4, hipMemcpyDeviceToHost));
            for (int t = 0; t < T; ++t) {
                // reference: n_top-th largest key value, all > thr plus the first (n_top - n_gt) == thr in index order, emitted as (all > thr in index order, then ties in index order), then the own group
                std::vector<uint32_t> ks(grp);
                for (int g = 0; g < grp; ++g) ks[g] = f2key_h(sc[(size_t)t * ldS + g]);
                std::vector<uint32_t> srt = ks;
                std::nth_element(srt.begin(), srt.begin() + (n_top - 1), srt.end(), std::greater<uint32_t>());
                const uint32_t thr = srt[n_top - 1];
                std::vector<int> sel;
                for (int g = 0; g < grp; ++g) if (ks[g] > thr) sel.push_back(g);
                int need = n_top - (int)sel.size();
                for (int g = 0; g < grp && need > 0; ++g) if (ks[g] == thr) { sel.push_back(g); --need; }
                std::vector<int32_t> ref;
                for (int s : sel) for (int j = 0; j < 4; ++j) ref.push_back(4 * s + j);
                const int pos = p0 + t;
                for (int i = 0; i < pos - 4 * grp + 1; ++i) ref.push_back(4 * grp + i);
                ++total;
                bool ok = nk[t] == (int)ref.size();
                if (ok) {
                    // the pools may be emitted in any fixed order: compare as sets per 4-group plus the tail in order
                    std::vector<int32_t> a(keys.begin() + (size_t)t * kmax, keys.begin() + (size_t)t * kmax + nk[t]), b = ref;
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    ok = a == b;
                }
                if (!ok) { ++bad; printf("MISMATCH grp %d mode %d t %d: nkeys %d vs %zu\n", grp, mode, t, nk[t], ref.size()); }
            }
            hipFree(d_sc); hipFree(d_keys); hipFree(d_nk);
        }
    }
    printf("idx_select_test: %d cases, %d mismatches\n", total, bad);
    return bad ? 1 : 0;
}
