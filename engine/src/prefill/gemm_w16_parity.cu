// src/prefill/gemm_w16_parity.cu - the FP16 weight cache (STRATA_PF_W16_CACHE_MIB) against the uncached Gemm::native, bitwise
// (GPU, synthetic weights, no model).
//
// gemm_dequant_side_parity runs Gemm::native twice, with the cache and without the side-stream scratch, and its scratch (10.5 M elements) is
// smaller than three of its eight weights, so those take the SLICED path (several GEMMs of N/k rows) uncached but ONE full-N GEMM cached:
// two different cuBLAS calls that need not round alike.  The engine's scratch is GEMM_SCRATCH = 32 M elements (src/prefill/prefill.cpp) and every
// dense projection of Qwen3.8-Flash-Next (the largest is attn_q, 12288 x 2560 = 31.5 M) fits it, so in the engine the cached and the uncached
// forms are the same GEMM on the same FP16 values.  This test runs exactly that: the engine's scratch size, the engine's dense shapes and types
// (Q8_0 on IQ3_S, Q4_K / Q5_K / Q6_K on the Unsloth sets), several token counts, with the cache off (STRATA_PF_W16_CACHE_MIB=0) and on
// ("auto"), each cached weight used twice.  Every output must be identical bit for bit.  A weight larger than the scratch (the 248320 x 2560
// output head) is reported separately and is NOT part of the verdict: it never goes through Gemm::native in the prompt path.
// Exit 77 without a CUDA device.
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
__global__ void fill_f16(__half* p, int64_t n, unsigned seed, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}
bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
// one quantization type: ggml type id, elements and bytes per block, offsets of the FP16 scales inside a block
struct QType { int type, elems, bytes; int at[2]; const char* name; };
struct Shape { int qt; int64_t N, K; const char* what; bool production; };
struct Weight { Shape s; void* dev = nullptr; };
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no CUDA device (skipped)\n"); return 77; }
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    const int64_t scratch_elems = 32ll << 20;   // GEMM_SCRATCH in src/prefill/prefill.cpp
    const QType qts[] = {{8, 32, 34, {0, -1}, "Q8_0"}, {12, 256, 144, {0, 2}, "Q4_K"}, {13, 256, 176, {0, 2}, "Q5_K"}, {14, 256, 210, {208, -1}, "Q6_K"}};
    // the dense projections of Qwen3.8-Flash-Next (index of the IQ3_S pack: attn_q 12288 x 2560, attn_qkv 10240, attn_gate 6144, ssm_out / attn_output
    // 2560 x 6144, the shared expert 640, attn_k / attn_v 512), in each type; the 248320-row head last (larger than the scratch)
    std::vector<Shape> shapes;
    for (int q = 0; q < 4; ++q) {
        shapes.push_back({q, 12288, 2560, "attn_q", true});
        shapes.push_back({q, 10240, 2560, "attn_qkv", true});
        shapes.push_back({q, 6144, 2560, "attn_gate", true});
        shapes.push_back({q, 2560, 6144, "ssm_out", true});
        shapes.push_back({q, 640, 2560, "shexp_gate", true});
        shapes.push_back({q, 2560, 640, "shexp_down", true});
        shapes.push_back({q, 512, 2560, "attn_k", true});
    }
    shapes.push_back({0, 248320, 2560, "output head (bigger than the scratch)", false});
    std::mt19937 rng(20261103u);
    std::vector<Weight> ws;
    for (const Shape& s : shapes) {
        const QType& q = qts[s.qt];
        if (s.K % q.elems != 0) continue;
        const size_t blocks = (size_t) (s.N * s.K / q.elems), bytes = blocks * (size_t) q.bytes;
        std::vector<uint8_t> h(bytes);
        for (auto& b : h) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < blocks; ++k)
            for (int at : q.at)
                if (at >= 0) { const uint16_t v = sane_half(rng); std::memcpy(&h[k * (size_t) q.bytes + (size_t) at], &v, 2); }
        Weight w; w.s = s;
        if (!ok(cudaMalloc(&w.dev, bytes), "W") || !ok(cudaMemcpy(w.dev, h.data(), bytes, cudaMemcpyHostToDevice), "W copy")) return 1;
        ws.push_back(w);
    }
    int64_t max_k = 0, max_n = 0;
    for (const Weight& w : ws) { if (w.s.K > max_k) max_k = w.s.K; if (w.s.N > max_n) max_n = w.s.N; }
    const int64_t Ts[] = {16, 64, 512, 4096};
    const int64_t Tmax = 4096;
    uint16_t* X = nullptr;
    float* Y[2] = {nullptr, nullptr};
    if (!ok(cudaMalloc((void**) &X, (size_t) Tmax * max_k * 2), "X") || !ok(cudaMalloc((void**) &Y[0], (size_t) Tmax * max_n * 4), "Y0") ||
        !ok(cudaMalloc((void**) &Y[1], (size_t) Tmax * max_n * 4), "Y1"))
        return 1;
    fill_f16<<<(unsigned) ((Tmax * max_k + 255) / 256), 256, 0, st>>>((__half*) X, Tmax * max_k, 17u, 2.0f);
    if (!ok(cudaStreamSynchronize(st), "fill")) return 1;

    int bad = 0, bad_other = 0, checked = 0;
    for (const Weight& w : ws) {
        for (const int64_t T : Ts) {
            std::vector<uint8_t> out[2];
            for (int on = 0; on < 2; ++on) {
                setenv("STRATA_PF_W16_CACHE_MIB", on ? "auto" : "0", 1);
                strata::prefill::Gemm gemm;          // a fresh one reads the variable at its first native()
                std::string err;
                if (!gemm.init(st, scratch_elems, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
                cudaMemset(Y[on], 0xff, (size_t) T * max_n * 4);
                for (int rep = 0; rep < (on ? 2 : 1); ++rep)   // cached: the first call fills the cache, the second reads it
                    gemm.native(X, qts[w.s.qt].type, w.dev, Y[on], T, w.s.N, w.s.K, max_n, 0.0f);
                if (!ok(cudaStreamSynchronize(st), "run") || !ok(cudaDeviceSynchronize(), "device")) return 1;
                out[on].resize((size_t) T * max_n * 4);
                cudaMemcpy(out[on].data(), Y[on], out[on].size(), cudaMemcpyDeviceToHost);
            }
            long long diff = 0;
            for (int64_t r = 0; r < T; ++r)
                if (std::memcmp(&out[0][(size_t) r * max_n * 4], &out[1][(size_t) r * max_n * 4], (size_t) w.s.N * 4) != 0) ++diff;
            const bool fits = w.s.N * w.s.K <= scratch_elems;
            if (w.s.production) ++checked;
            if (diff || !w.s.production)
                std::printf("  %-5s %-22s N %6lld K %4lld T %5lld%s: %s\n", qts[w.s.qt].name, w.s.what, (long long) w.s.N, (long long) w.s.K, (long long) T,
                            fits ? "" : " (sliced when uncached)", diff ? "*** DIFFERS ***" : "bitwise equal");
            if (diff) { if (w.s.production) ++bad; else ++bad_other; }
        }
    }
    std::printf("gemm_w16_parity: %d production (shape, type, T) combinations checked, %d differ; head (not part of the verdict) %s\n", checked, bad,
                bad_other ? "differs" : "equal");
    std::printf("gemm_w16_parity: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
