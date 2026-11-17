// src/prefill/qsa_nr_parity.cu - the QSA layer's fused norm + rope (rms_rope_rows, STRATA_QSA_FUSE_NR) against the chain it
// replaces (split_q + rms_rows + rope), compared BIT FOR BIT over the whole output, for the three uses in the prompt path:
//   query   Qf [T, 24, 512] -> q [T, 24, 256]            (split, norm, rotate)
//   key     Kc [T, 2, 256] in place                        (norm, rotate)
//   indexer q_idx [T, 4, 128] with row stride 512, in place (norm, rotate)
// Every output element is checked (not a sample), the output buffers are poisoned with NaN first, and T runs through
// 1 .. 8191 with several start positions, so a tile or row that is not written shows up.  Exit 77 when there is no GPU.
//   qsa_nr_parity           the checks;   qsa_nr_parity --bench   and the timings at T = 2048 / 8191
#include "strata/kernels/rope_scaling.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) std::exit(2);
    return p;
}
template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
void fill(std::vector<float>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    for (float& x : v) x = nd(rng);
}
size_t diffs(const std::vector<float>& a, const std::vector<float>& b, size_t& first) {
    size_t n = 0;
    first = (size_t) -1;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) {
            if (first == (size_t) -1) first = i;
            ++n;
        }
    }
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace strata::prefill;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const strata::kernels::RopeScaling& rs = strata::kernels::rope_scaling();
    const int64_t TMAX = 8191;
    std::mt19937 rng(12345);
    std::vector<float> hqf((size_t) TMAX * 24 * 512), hk((size_t) TMAX * 512), hqi((size_t) TMAX * 512), wq(256), wk(256), wi(128);
    fill(hqf, rng, 2.0f);
    fill(hk, rng, 3.0f);
    fill(hqi, rng, 1.5f);
    for (float& x : wq) x = 0.5f + std::abs(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    for (float& x : wk) x = 0.5f + std::abs(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    for (float& x : wi) x = 0.5f + std::abs(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    // a few rows with large and tiny magnitudes, and an all-zero row (rsqrt(eps))
    for (int d = 0; d < 256; ++d) { hqf[(size_t) 3 * 24 * 512 + 512 * 5 + d] = 0.0f; hqf[(size_t) 4 * 24 * 512 + d] *= 1e4f; hk[(size_t) 7 * 512 + d] *= 1e-6f; }
    float *dqf = dalloc<float>(hqf.size()), *dk0 = dalloc<float>(hk.size()), *dk1 = dalloc<float>(hk.size()),
          *dq0 = dalloc<float>((size_t) TMAX * 24 * 256), *dq1 = dalloc<float>((size_t) TMAX * 24 * 256),
          *di0 = dalloc<float>(hqi.size()), *di1 = dalloc<float>(hqi.size()), *dwq = dalloc<float>(256), *dwk = dalloc<float>(256),
          *dwi = dalloc<float>(128);
    ck(cudaMemcpy(dqf, hqf.data(), hqf.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dwq, wq.data(), 1024, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dwk, wk.data(), 1024, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dwi, wi.data(), 512, cudaMemcpyHostToDevice), "up");
    const float eps = 1e-6f;
    int fails = 0;
    const int64_t Ts[] = {1, 2, 7, 31, 33, 64, 257, 1000, 2048, 4093, 8191};
    const int64_t P0s[] = {0, 5, 4093, 131071};
    for (int64_t T : Ts) {
        for (int64_t p0 : P0s) {
            // old chain
            ck(cudaMemset(dq0, 0xff, (size_t) T * 24 * 256 * 4), "poison");
            ck(cudaMemset(dq1, 0xff, (size_t) T * 24 * 256 * 4), "poison");
            ck(cudaMemcpy(dk0, hk.data(), (size_t) T * 512 * 4, cudaMemcpyHostToDevice), "up");
            ck(cudaMemcpy(dk1, hk.data(), (size_t) T * 512 * 4, cudaMemcpyHostToDevice), "up");
            ck(cudaMemcpy(di0, hqi.data(), (size_t) T * 512 * 4, cudaMemcpyHostToDevice), "up");
            ck(cudaMemcpy(di1, hqi.data(), (size_t) T * 512 * 4, cudaMemcpyHostToDevice), "up");
            split_q(dqf, dq0, T, nullptr);
            rms_rows(dq0, dwq, T * 24, 256, 256, eps, nullptr);
            rope(dq0, T, 24, 256, 6144, p0, rs, nullptr);
            rms_rows(dk0, dwk, T * 2, 256, 256, eps, nullptr);
            rope(dk0, T, 2, 256, 512, p0, rs, nullptr);
            rms_rows(di0, dwi, T * 4, 128, 128, eps, nullptr);
            rope(di0, T, 4, 128, 512, p0, rs, nullptr);
            // fused
            rms_rope_rows(dqf, 24 * 512, 512, dq1, 6144, dwq, T, 24, 256, eps, p0, rs, nullptr);
            rms_rope_rows(dk1, 512, 256, dk1, 512, dwk, T, 2, 256, eps, p0, rs, nullptr);
            rms_rope_rows(di1, 512, 128, di1, 512, dwi, T, 4, 128, eps, p0, rs, nullptr);
            if (!ck(cudaDeviceSynchronize(), "sync")) return 1;
            size_t f0 = 0, f1 = 0, f2 = 0;
            const size_t nq = diffs(download(dq0, (size_t) T * 24 * 256), download(dq1, (size_t) T * 24 * 256), f0);
            const size_t nk = diffs(download(dk0, (size_t) T * 512), download(dk1, (size_t) T * 512), f1);
            const size_t ni = diffs(download(di0, (size_t) T * 512), download(di1, (size_t) T * 512), f2);
            // the poison must be gone everywhere in the fused query output
            const std::vector<float> q1 = download(dq1, (size_t) T * 24 * 256);
            size_t nan = 0;
            for (float x : q1) nan += x != x;
            const bool ok = nq == 0 && nk == 0 && ni == 0 && nan == 0;
            if (!ok) ++fails;
            std::printf("T %5lld p0 %6lld  q diff %zu (first %lld)  k diff %zu (first %lld)  idx diff %zu (first %lld)  NaN %zu  %s\n",
                        (long long) T, (long long) p0, nq, (long long) f0, nk, (long long) f1, ni, (long long) f2, nan, ok ? "OK" : "FAIL");
        }
    }
    if (bench) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int64_t T : {(int64_t) 2048, (int64_t) 8191}) {
            auto time = [&](auto&& fn) {
                for (int i = 0; i < 3; ++i) fn();
                cudaEventRecord(a);
                for (int i = 0; i < 20; ++i) fn();
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                return ms / 20.0f;
            };
            const float old_ms = time([&] {
                split_q(dqf, dq0, T, nullptr);
                rms_rows(dq0, dwq, T * 24, 256, 256, eps, nullptr);
                rope(dq0, T, 24, 256, 6144, 100, rs, nullptr);
                rms_rows(dk0, dwk, T * 2, 256, 256, eps, nullptr);
                rope(dk0, T, 2, 256, 512, 100, rs, nullptr);
                rms_rows(di0, dwi, T * 4, 128, 128, eps, nullptr);
                rope(di0, T, 4, 128, 512, 100, rs, nullptr);
            });
            const float new_ms = time([&] {
                rms_rope_rows(dqf, 24 * 512, 512, dq1, 6144, dwq, T, 24, 256, eps, 100, rs, nullptr);
                rms_rope_rows(dk1, 512, 256, dk1, 512, dwk, T, 2, 256, eps, 100, rs, nullptr);
                rms_rope_rows(di1, 512, 128, di1, 512, dwi, T, 4, 128, eps, 100, rs, nullptr);
            });
            std::printf("bench T %lld: chain %.3f ms, fused %.3f ms (x%.2f) per QSA layer\n", (long long) T, old_ms, new_ms,
                        old_ms / new_ms);
        }
    }
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
