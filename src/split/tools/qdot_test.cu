// The split engine's decode dots (src/qdot.cuh) against llama.cpp's own (ggml-cuda vecdotq.cuh, the vendored third_party/strata-ggml pin) on random
// blocks: every one of the 11 formats the engine runs, every block of a random matrix, every (kbx, iqs) the decode kernels call with; the floats must
// be bit for bit the same. Prints one line per format and exits 1 on any difference.
//   split_qdot_test [blocks per format]      (CUDA: built with the split; HIP: the same sources with -DFNX_HIP=1 GGML_USE_HIP=1)
//
// ORACLE NOTE (own-stack plan): this is the one file of the split engine that still includes ggml-cuda (vecdotq.cuh, as the reference). When the
// vendored ggml tree is removed, this test goes with it: by then qdot.cuh has been proved equal on CUDA and HIP and the engine's identity runs
// guard it (the proposal in phaseA-tests/par/RESULTS-OWN-4S.md; frozen reference vectors are the alternative if the coordinator wants a test left).
// That alternative is taken: split_qdot_test --write-golden <file> writes the digests of qdot_golden.cuh's inputs (only when every dot equals ggml's),
// --regen-golden <file> checks the stored file comes out byte for byte again, and split_qdot_golden (qdot_golden.cu, no ggml) checks qdot.cuh against
// the file: that test stays when this one goes.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if FNX_HIP
// as the engine had it before qdot.cuh: ggml's HIP vendor header defines these three itself
#undef __vsubss4
#undef __vsub4
#undef __vcmpne4
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#endif
#include "common.cuh"
#include "vecdotq.cuh"
#if FNX_HIP
#undef cudaFuncSetAttribute
#endif

#include "qdot.cuh"
#include "qdot_golden.cuh"

namespace {

using namespace fnx;

#define FNX_QD_PAIR(T, NAME, VDRG)                                                                                                   \
    template <> struct Pair<T> {                                                                                                      \
        static constexpr int vdr = VDRG;                                                                                              \
        __device__ static float ggml(const void* v, const block_q8_1* y, int k, int i) { return ::NAME(v, y, k, i); }                \
        __device__ static float ours(const void* v, const block_q8_1* y, int k, int i) { return fnx::NAME(v, y, k, i); }             \
    };

template <GgmlType T> struct Pair;
FNX_QD_PAIR(T_Q4_0, vec_dot_q4_0_q8_1, VDR_Q4_0_Q8_1_MMVQ)
FNX_QD_PAIR(T_Q5_1, vec_dot_q5_1_q8_1, VDR_Q5_1_Q8_1_MMVQ)
FNX_QD_PAIR(T_Q8_0, vec_dot_q8_0_q8_1, VDR_Q8_0_Q8_1_MMVQ)
FNX_QD_PAIR(T_Q4_K, vec_dot_q4_K_q8_1, VDR_Q4_K_Q8_1_MMVQ)
FNX_QD_PAIR(T_Q5_K, vec_dot_q5_K_q8_1, VDR_Q5_K_Q8_1_MMVQ)
FNX_QD_PAIR(T_Q6_K, vec_dot_q6_K_q8_1, VDR_Q6_K_Q8_1_MMVQ)
FNX_QD_PAIR(T_IQ4_NL, vec_dot_iq4_nl_q8_1, VDR_IQ4_NL_Q8_1_MMVQ)
FNX_QD_PAIR(T_IQ4_XS, vec_dot_iq4_xs_q8_1, VDR_IQ4_XS_Q8_1_MMVQ)
FNX_QD_PAIR(T_IQ3_S, vec_dot_iq3_s_q8_1, VDR_IQ3_S_Q8_1_MMVQ)
FNX_QD_PAIR(T_IQ3_XXS, vec_dot_iq3_xxs_q8_1, VDR_IQ3_XXS_Q8_1_MMVQ)
FNX_QD_PAIR(T_IQ2_S, vec_dot_iq2_s_q8_1, VDR_IQ2_S_Q8_1_MMVQ)

// one thread per (weight block, lane): the call the decode kernels make (kernels.cu k_mmvq: y + kbx * (qk / QK8_1), iqs = vdr * lane)
template <GgmlType T>
__global__ void k_cmp(const char* w, const block_q8_1* y, int nblocks, unsigned* bad, unsigned* checked, unsigned* trait_bad) {
    constexpr int qk = qtraits<T>::qk, qi = qtraits<T>::qi, per = qi / Pair<T>::vdr;
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (qk != ggml_cuda_type_traits<(ggml_type) T>::qk || qi != ggml_cuda_type_traits<(ggml_type) T>::qi)) atomicAdd(trait_bad, 1u);
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nblocks * per) return;
    const int kbx = e / per, iqs = Pair<T>::vdr * (e % per);
    const block_q8_1* yb = y + (size_t) kbx * (qk / QK8_1);
    const float a = Pair<T>::ggml(w, yb, kbx, iqs), b = Pair<T>::ours(w, yb, kbx, iqs);
    atomicAdd(checked, 1u);
    if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(bad, 1u);
}

__half finite_half(std::mt19937& rng) { return __float2half((float) ((int) (rng() % 2001) - 1000) / 400.0f); }

// B: block bytes; h16: byte offsets of the block's fp16 fields (made finite: random bytes can be NaN / inf there)
template <GgmlType T>
bool run(const char* name, int B, std::vector<int> h16, int nblocks, unsigned seed, int our_vdr) {
    constexpr int qk = qtraits<T>::qk, qi = qtraits<T>::qi;
    if (our_vdr != Pair<T>::vdr) {
        std::printf("%s: VDR %d, ggml's %d  <-- MISMATCH\n", name, our_vdr, Pair<T>::vdr);
        return false;
    }
    std::mt19937 rng(seed);
    std::vector<unsigned char> w((size_t) nblocks * B);
    for (auto& c : w) c = (unsigned char) rng();
    for (int k = 0; k < nblocks; ++k)
        for (int off : h16) {
            const __half h = finite_half(rng);
            std::memcpy(&w[(size_t) k * B + off], &h, 2);
        }
    std::vector<block_q8_1> y((size_t) nblocks * (qk / QK8_1));
    for (auto& b : y) {
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (int) (rng() % 255 - 127);
        // s drawn first, then d (qdot_golden.cuh make_inputs: the same draws, a fixed order on every compiler)
        const float s = (float) ((int) (rng() % 200) - 100);
        const float d = (float) (rng() % 1000) / 500.0f;
        b.ds = make_half2(__float2half(d), __float2half(s));
    }
    char* dw;
    block_q8_1* dy;
    unsigned *dbad, *dchk, *dtr;
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dy, y.size() * sizeof(block_q8_1));
    cudaMalloc(&dbad, 4);
    cudaMalloc(&dchk, 4);
    cudaMalloc(&dtr, 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);
    cudaMemset(dbad, 0, 4);
    cudaMemset(dchk, 0, 4);
    cudaMemset(dtr, 0, 4);
    const int per = qi / Pair<T>::vdr;
    k_cmp<T><<<(nblocks * per + 255) / 256, 256>>>(dw, dy, nblocks, dbad, dchk, dtr);
    const cudaError_t e = cudaDeviceSynchronize();
    unsigned bad = 0, chk = 0, tr = 0;
    cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&chk, dchk, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&tr, dtr, 4, cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dy); cudaFree(dbad); cudaFree(dchk); cudaFree(dtr);
    const bool ok = e == cudaSuccess && bad == 0 && tr == 0 && chk == (unsigned) nblocks * per;
    std::printf("%-8s %9u dots compared, %u differ, traits %s%s%s\n", name, chk, bad, tr ? "DIFFER" : "same", e == cudaSuccess ? "" : "  cuda error: ",
                e == cudaSuccess ? (ok ? "" : "  <-- MISMATCH") : cudaGetErrorString(e));
    return ok;
}

// --write-golden / --regen-golden: ggml's dots on qdot_golden.cuh's inputs, in its order (the oracle side of the stored digests)
template <GgmlType T> __global__ void k_ggml(const char* w, const block_q8_1* y, int nblocks, float* out) {
    constexpr int qk = qtraits<T>::qk, per = qtraits<T>::qi / Pair<T>::vdr;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nblocks * per) return;
    const int kbx = e / per, iqs = Pair<T>::vdr * (e % per);
    out[e] = Pair<T>::ggml(w, y + (size_t) kbx * (qk / QK8_1), kbx, iqs);
}
struct Writer {
    std::string text;
    int differ = 0, lines = 0;
    template <GgmlType T> bool operator()(const fnx_golden::Format& f, const fnx_golden::Inputs& in) {
        static_assert(Pair<T>::vdr == fnx_golden::Dot<T>::vdr, "the same VDR on both sides");
        const std::vector<float> ours = fnx_golden::run_ours<T>(in, f.nblocks);
        std::vector<float> ref(ours.size());
        char* dw = nullptr;
        block_q8_1* dy = nullptr;
        float* dout = nullptr;
        bool ok = !ours.empty() && cudaMalloc(&dw, in.w.size()) == cudaSuccess &&
                  cudaMalloc(&dy, in.y.size() * sizeof(block_q8_1)) == cudaSuccess && cudaMalloc(&dout, ref.size() * 4) == cudaSuccess;
        if (ok) {
            cudaMemcpy(dw, in.w.data(), in.w.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dy, in.y.data(), in.y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);
            k_ggml<T><<<((int) ref.size() + 255) / 256, 256>>>(dw, dy, f.nblocks, dout);
            ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        cudaFree(dw); cudaFree(dy); cudaFree(dout);
        if (!ok) { std::printf("%-8s cuda error\n", f.name); ++differ; return false; }
        int bad = 0;
        for (size_t i = 0; i < ours.size(); ++i) bad += std::memcmp(&ours[i], &ref[i], 4) != 0;
        differ += bad;
        const std::vector<fnx_golden::Line> g = fnx_golden::group_lines("cuda", f.name, in, ours);
        for (const fnx_golden::Line& l : g) text += fnx_golden::line_text(l) + "\n";
        lines += (int) g.size();
        std::printf("%-8s %7zu dots against ggml: %d differ, %zu groups\n", f.name, ours.size(), bad, g.size());
        return bad == 0;
    }
};
int golden(const char* mode, const char* path) {
    Writer wr;
    wr.text = fnx_golden::kHeader;
    const bool same = fnx_golden::for_each_format(wr);
    if (!same || wr.differ) { std::printf("qdot.cuh differs from ggml on %d dots: nothing written\n", wr.differ); return 1; }
    if (std::strcmp(mode, "--write-golden") == 0) {
        FILE* f = std::fopen(path, "wb");
        if (!f || std::fwrite(wr.text.data(), 1, wr.text.size(), f) != wr.text.size()) { std::printf("cannot write %s\n", path); return 1; }
        std::fclose(f);
        std::printf("wrote %s: %d groups, %zu bytes\n", path, wr.lines, wr.text.size());
        return 0;
    }
    // --regen-golden: the cuda lines and the header must come out byte for byte as stored (lines of other families are kept as they are)
    std::string stored;
    if (FILE* f = std::fopen(path, "rb")) {
        char b[4096];
        size_t k;
        while ((k = std::fread(b, 1, sizeof b, f)) > 0) stored.append(b, k);
        std::fclose(f);
    } else { std::printf("cannot read %s\n", path); return 1; }
    std::string kept;   // the stored text without other families' lines
    for (size_t p = 0; p < stored.size();) {
        size_t e = stored.find('\n', p);
        e = e == std::string::npos ? stored.size() : e + 1;
        const std::string ln = stored.substr(p, e - p);
        if (ln.rfind("hip ", 0) != 0) kept += ln;
        p = e;
    }
    const bool eq = kept == wr.text;
    std::printf("regenerated %d groups, %zu bytes: %s the stored file (%zu bytes without other families' lines)\n", wr.lines, wr.text.size(),
                eq ? "byte for byte equal to" : "DIFFERENT FROM", kept.size());
    return eq ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // split_qdot_test --write-golden <file> | --regen-golden <file>: the digests of qdot_golden.cuh (see there); otherwise the comparison below
    if (argc > 2 && (std::strcmp(argv[1], "--write-golden") == 0 || std::strcmp(argv[1], "--regen-golden") == 0)) return golden(argv[1], argv[2]);
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 18;
    bool ok = true;
    ok &= run<T_Q4_0>("Q4_0", sizeof(block_q4_0), {0}, n * 8, 1, VDR_Q4_0);
    ok &= run<T_Q5_1>("Q5_1", sizeof(block_q5_1), {0, 2}, n * 8, 2, VDR_Q5_1);
    ok &= run<T_Q8_0>("Q8_0", sizeof(block_q8_0), {0}, n * 8, 3, VDR_Q8_0);
    ok &= run<T_Q4_K>("Q4_K", sizeof(block_q4_K), {0, 2}, n, 4, VDR_Q4_K);
    ok &= run<T_Q5_K>("Q5_K", sizeof(block_q5_K), {0, 2}, n, 5, VDR_Q5_K);
    ok &= run<T_Q6_K>("Q6_K", sizeof(block_q6_K), {(int) sizeof(block_q6_K) - 2}, n, 6, VDR_Q6_K);
    ok &= run<T_IQ4_NL>("IQ4_NL", sizeof(block_iq4_nl), {0}, n * 8, 7, VDR_IQ4_NL);
    ok &= run<T_IQ4_XS>("IQ4_XS", sizeof(block_iq4_xs), {0}, n, 8, VDR_IQ4_XS);
    ok &= run<T_IQ3_S>("IQ3_S", sizeof(block_iq3_s), {0}, n, 9, VDR_IQ3_S);
    ok &= run<T_IQ3_XXS>("IQ3_XXS", sizeof(block_iq3_xxs), {0}, n, 10, VDR_IQ3_XXS);
    ok &= run<T_IQ2_S>("IQ2_S", sizeof(block_iq2_s), {0}, n, 11, VDR_IQ2_S);
    std::printf(ok ? "qdot_test: PASS\n" : "qdot_test: FAIL\n");
    return ok ? 0 : 1;
}
