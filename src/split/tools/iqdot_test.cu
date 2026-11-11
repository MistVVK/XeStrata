// HIP: the i-quant decode dots of src/iqdot_hip.cuh against src/qdot.cuh's vec_dot_iq3_s / iq3_xxs / iq2_s_q8_1 (ggml's arithmetic; tools/qdot_test.cu ties them to vecdotq.cuh) on random blocks (random weight bytes, random q8_1 rows, every
// iqs the kernels use): the floats must match bit for bit. Prints one line per type and exits 1 on a mismatch.
//   build: the same compile line as src/kernels.cu's parts (-DFNX_HIP=1 ...), see phaseA-tests/round0211/r9700split/iqdot_build.sh
#include "kernels.hpp"
#include "env.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "qdot.cuh"
#include "iqdot_hip.cuh"

namespace {

template <int T> struct Dots;
template <> struct Dots<0> {   // IQ3_S
    static constexpr int BYTES = sizeof(block_iq3_s);
    __device__ static float ref(const void* v, const block_q8_1* y, int k, int i) { return fnx::vec_dot_iq3_s_q8_1(v, y, k, i); }
    __device__ static float mine(const void* v, const block_q8_1* y, int k, int i) { return fnx::fnx_vec_dot_iq3_s_q8_1(v, y, k, i); }
};
template <> struct Dots<1> {   // IQ3_XXS
    static constexpr int BYTES = sizeof(block_iq3_xxs);
    __device__ static float ref(const void* v, const block_q8_1* y, int k, int i) { return fnx::vec_dot_iq3_xxs_q8_1(v, y, k, i); }
    __device__ static float mine(const void* v, const block_q8_1* y, int k, int i) { return fnx::fnx_vec_dot_iq3_xxs_q8_1(v, y, k, i); }
};
template <> struct Dots<2> {   // IQ2_S
    static constexpr int BYTES = sizeof(block_iq2_s);
    __device__ static float ref(const void* v, const block_q8_1* y, int k, int i) { return fnx::vec_dot_iq2_s_q8_1(v, y, k, i); }
    __device__ static float mine(const void* v, const block_q8_1* y, int k, int i) { return fnx::fnx_vec_dot_iq2_s_q8_1(v, y, k, i); }
};

// one thread per (block, iqs): the two dots against q8_1 block y[block * 8 + iqs / 2... ] (the dot reads bq8_1[iqs / 2], so 8 q8_1 blocks per weight block)
template <int T>
__global__ void k_cmp(const char* w, const block_q8_1* y, int nblocks, unsigned* bad, unsigned* checked) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nblocks * 8) return;
    const int kbx = e / 8, iqs = (e % 8) * 2;
    const float a = Dots<T>::ref(w, y + (size_t) kbx * 8, kbx, iqs), b = Dots<T>::mine(w, y + (size_t) kbx * 8, kbx, iqs);
    atomicAdd(checked, 1u);
    if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(bad, 1u);
}

template <int T>
bool run(const char* name, int nblocks, unsigned seed) {
    constexpr int B = Dots<T>::BYTES;
    std::mt19937 rng(seed);
    std::vector<unsigned char> w((size_t) nblocks * B);
    for (auto& c : w) c = (unsigned char) rng();
    for (int k = 0; k < nblocks; ++k) {   // d: a finite half of either sign
        const __half h = __float2half((float) ((int) (rng() % 2001) - 1000) / 400.0f);
        std::memcpy(&w[(size_t) k * B], &h, 2);
    }
    std::vector<block_q8_1> y((size_t) nblocks * 8);
    for (auto& b : y) {
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (int) (rng() % 255 - 127);
        // s drawn first, then d (as qdot_golden.cuh make_inputs: a fixed order on every compiler)
        const float s = (float) ((int) (rng() % 200) - 100);
        const float d = (float) (rng() % 1000) / 500.0f;
        b.ds = make_half2(__float2half(d), __float2half(s));
    }
    char* dw;
    block_q8_1* dy;
    unsigned *dbad, *dchk;
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dy, y.size() * sizeof(block_q8_1));
    cudaMalloc(&dbad, 4);
    cudaMalloc(&dchk, 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);
    cudaMemset(dbad, 0, 4);
    cudaMemset(dchk, 0, 4);
    k_cmp<T><<<(nblocks * 8 + 255) / 256, 256>>>(dw, dy, nblocks, dbad, dchk);
    cudaDeviceSynchronize();
    unsigned bad = 0, chk = 0;
    cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&chk, dchk, 4, cudaMemcpyDeviceToHost);
    std::printf("%s: %u dots compared, %u differ%s\n", name, chk, bad, bad ? "  <-- MISMATCH" : "");
    cudaFree(dw); cudaFree(dy); cudaFree(dbad); cudaFree(dchk);
    return bad == 0 && chk == (unsigned) nblocks * 8;
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    bool ok = run<0>("IQ3_S", n, 1) & run<1>("IQ3_XXS", n, 2) & run<2>("IQ2_S", n, 3);
    std::printf(ok ? "iqdot_test: PASS\n" : "iqdot_test: FAIL\n");
    return ok ? 0 : 1;
}
