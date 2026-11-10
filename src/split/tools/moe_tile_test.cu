// The prompt kernel's weight tiles (src/moe_mma.cuh, src/moe_mma_iq.cuh) against the file: for one expert of each routed-expert type a
// GGUF holds, runs the tiles' fetch() and put() over the whole matrix on the GPU and writes the f16 values they put into shared memory.
// tests/split/tile_check.py compares those with gguf-py's dequantization of the same expert (rounded once to f16).
//   bin/moe_tile_test MODEL.gguf OUTDIR [expert ...]      writes OUTDIR/<tensor>.e<expert>.f16 and prints "name type nrows ncols expert"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "qdot.cuh"   // block layouts and codebook tables

namespace fnx {
namespace {

__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }

#include "moe_w.cuh"
#include "moe_mma.cuh"
#include "moe_mma_iq.cuh"

template <int TYPE>
__global__ void __launch_bounds__(MM_THREADS) k_tiles(const char* __restrict__ we, size_t row_a, int nrows, int ncols, __half* __restrict__ out) {
    extern __shared__ __align__(32) __half smem[];
    constexpr int NP = MM_BM / MM_PASS;
    const int row0 = blockIdx.x * MM_BM;
    for (int k0 = 0; k0 < ncols; k0 += MM_KC) {
        const int kc = min(MM_KC, ncols - k0);
        MmTile<TYPE> wt[NP];
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].fetch(we, row_a, nrows, row0 + p * MM_PASS, k0, kc);
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].put(k0, kc, smem + p * MM_PASS * MM_LD);
        __syncthreads();
        for (int e = threadIdx.x; e < MM_BM * kc; e += MM_THREADS) {
            const int r = e / kc, c = e % kc, row = row0 + r;
            if (row < nrows) out[(size_t) row * ncols + k0 + c] = smem[r * MM_LD + c];
        }
        __syncthreads();
    }
}

template <int TYPE>
void run(const char* d_we, size_t row_a, int nrows, int ncols, __half* d_out) {
    k_tiles<TYPE><<<(nrows + MM_BM - 1) / MM_BM, MM_THREADS, MM_BM * MM_LD * sizeof(__half)>>>(d_we, row_a, nrows, ncols, d_out);
}

}  // namespace
}  // namespace fnx

using namespace fnx;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moe_tile_test MODEL.gguf OUTDIR [expert ...]\n");
        return 2;
    }
    try {
        Gguf g(argv[1]);
        std::vector<int> experts;
        for (int i = 3; i < argc; ++i) experts.push_back(std::atoi(argv[i]));
        if (experts.empty()) experts = {5, 511};
        std::map<uint32_t, std::string> pick;   // the first routed-expert tensor of each type, gate/up first
        for (const auto& kv : g.tensors()) {
            const std::string& n = kv.first;
            if (n.find("_exps.weight") == std::string::npos) continue;
            const uint32_t ty = kv.second.type;
            if (!pick.count(ty)) pick[ty] = n;
        }
        // also a down matrix of each type that has both, as its K (640) is not a multiple of 256
        for (const auto& kv : g.tensors()) {
            const std::string& n = kv.first;
            if (n.find("ffn_down_exps.weight") == std::string::npos) continue;
            const std::string key = "down:" + std::to_string(kv.second.type);
            static std::map<std::string, bool> seen;
            if (!seen[key]) {
                seen[key] = true;
                pick[1000 + kv.second.type] = n;
            }
        }
        for (const auto& pk : pick) {
            const Tensor& t = g.t(pk.second);
            const int ncols = (int) t.ne[0], nrows = (int) t.ne[1];
            const size_t row_a = t.row_bytes();
            for (int e : experts) {
                if (e >= t.ne[2]) continue;
                const char* we = (const char*) t.data + (size_t) e * row_a * nrows;
                char* d_we;
                __half* d_out;
                if (cudaMalloc(&d_we, row_a * nrows) || cudaMalloc(&d_out, (size_t) nrows * ncols * 2)) throw std::runtime_error("cudaMalloc");
                cudaMemcpy(d_we, we, row_a * nrows, cudaMemcpyHostToDevice);
                cudaMemset(d_out, 0xFF, (size_t) nrows * ncols * 2);
                switch (t.type) {
                    case T_Q4_0: run<T_Q4_0>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_IQ4_NL: run<T_IQ4_NL>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_IQ4_XS: run<T_IQ4_XS>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_IQ3_S: run<T_IQ3_S>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_IQ3_XXS: run<T_IQ3_XXS>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_IQ2_S: run<T_IQ2_S>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_Q4_K: run<T_Q4_K>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_Q5_K: run<T_Q5_K>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_Q5_1: run<T_Q5_1>(d_we, row_a, nrows, ncols, d_out); break;
                    case T_Q8_0: run<T_Q8_0>(d_we, row_a, nrows, ncols, d_out); break;
                    default: std::printf("skip %s type %u\n", t.name.c_str(), t.type); cudaFree(d_we); cudaFree(d_out); continue;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(std::string("kernel: ") + cudaGetErrorString(cudaGetLastError()));
                std::vector<__half> h((size_t) nrows * ncols);
                cudaMemcpy(h.data(), d_out, h.size() * 2, cudaMemcpyDeviceToHost);
                cudaFree(d_we);
                cudaFree(d_out);
                const std::string path = std::string(argv[2]) + "/" + t.name + ".e" + std::to_string(e) + ".f16";
                FILE* fp = std::fopen(path.c_str(), "wb");
                if (!fp) throw std::runtime_error("cannot write " + path);
                std::fwrite(h.data(), 2, h.size(), fp);
                std::fclose(fp);
                std::printf("%s %u %d %d %d\n", t.name.c_str(), t.type, nrows, ncols, e);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "moe_tile_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
