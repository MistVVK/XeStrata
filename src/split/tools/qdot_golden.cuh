// The split engine's decode dots (src/qdot.cuh) against stored reference digests (src/split/tools/qdot_golden.txt): the inputs of every group are drawn
// again from fixed seeds (the same draws as qdot_test.cu's), the dots run on the card, and the FNV-1a 64-bit digest of the group's float bits must equal
// the stored one.  So the check needs no ggml: the file was written by qdot_test.cu --write-golden, which refuses to write unless every dot of qdot.cuh is
// bit for bit ggml-cuda's (vecdotq.cuh at the vendored pin) on the same inputs.
//
// File: '#' comment lines, then one line a group: <family> <format> <group> <dots> <input digest> <output digest> [% reason].  Family "cuda" holds every
// group; another toolchain family ("hip") has a line only where its bits differ from the cuda line, with the reason (measured ulp range) after '%'; a
// check under family F takes F's line where there is one, else the cuda line.  Shared by qdot_test.cu (writer, oracle build) and qdot_golden.cu (check).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "qdot.cuh"

namespace fnx_golden {

using namespace fnx;

constexpr int kGroupDots = 4096;   // dots a group (the last group of a format may be shorter)

template <GgmlType T> struct Dot;
#define FNX_GD_DOT(T, NAME, VDR)                                                                                           \
    template <> struct Dot<T> {                                                                                            \
        static constexpr int vdr = VDR;                                                                                    \
        __device__ static float ours(const void* v, const block_q8_1* y, int k, int i) { return fnx::NAME(v, y, k, i); } \
    };
FNX_GD_DOT(T_Q4_0, vec_dot_q4_0_q8_1, VDR_Q4_0)
FNX_GD_DOT(T_Q5_1, vec_dot_q5_1_q8_1, VDR_Q5_1)
FNX_GD_DOT(T_Q8_0, vec_dot_q8_0_q8_1, VDR_Q8_0)
FNX_GD_DOT(T_Q4_K, vec_dot_q4_K_q8_1, VDR_Q4_K)
FNX_GD_DOT(T_Q5_K, vec_dot_q5_K_q8_1, VDR_Q5_K)
FNX_GD_DOT(T_Q6_K, vec_dot_q6_K_q8_1, VDR_Q6_K)
FNX_GD_DOT(T_IQ4_NL, vec_dot_iq4_nl_q8_1, VDR_IQ4_NL)
FNX_GD_DOT(T_IQ4_XS, vec_dot_iq4_xs_q8_1, VDR_IQ4_XS)
FNX_GD_DOT(T_IQ3_S, vec_dot_iq3_s_q8_1, VDR_IQ3_S)
FNX_GD_DOT(T_IQ3_XXS, vec_dot_iq3_xxs_q8_1, VDR_IQ3_XXS)
FNX_GD_DOT(T_IQ2_S, vec_dot_iq2_s_q8_1, VDR_IQ2_S)
#undef FNX_GD_DOT

// one thread a dot, in qdot_test.cu's order: dot e = (weight block kbx = e / per, iqs = vdr (e % per)), the call the decode kernels make
template <GgmlType T> __global__ void k_ours(const char* w, const block_q8_1* y, int nblocks, float* out) {
    constexpr int qk = qtraits<T>::qk, per = qtraits<T>::qi / Dot<T>::vdr;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nblocks * per) return;
    const int kbx = e / per, iqs = Dot<T>::vdr * (e % per);
    out[e] = Dot<T>::ours(w, y + (size_t) kbx * (qk / QK8_1), kbx, iqs);
}

inline uint64_t fnv1a64(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// the inputs of one format: random weight bytes with finite fp16 fields, random q8_1 activations (qdot_test.cu's draws for the same seed and size)
struct Inputs {
    std::vector<unsigned char> w;
    std::vector<block_q8_1> y;
};
inline __half finite_half(std::mt19937& rng) { return __float2half((float) ((int) (rng() % 2001) - 1000) / 400.0f); }
inline Inputs make_inputs(int B, const std::vector<int>& h16, int nblocks, int qk, unsigned seed) {
    Inputs in;
    std::mt19937 rng(seed);
    in.w.resize((size_t) nblocks * B);
    for (auto& c : in.w) c = (unsigned char) rng();
    for (int k = 0; k < nblocks; ++k)
        for (int off : h16) {
            const __half h = finite_half(rng);
            std::memcpy(&in.w[(size_t) k * B + off], &h, 2);
        }
    in.y = std::vector<block_q8_1>((size_t) nblocks * (qk / QK8_1));   // (constructed at its size: block_q8_1 has no copy constructor, so no resize)
    for (auto& b : in.y) {
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (int) (rng() % 255 - 127);
        b.ds = make_half2(__float2half((float) (rng() % 1000) / 500.0f), __float2half((float) ((int) (rng() % 200) - 100)));
    }
    return in;
}

// the formats, their block bytes, fp16 fields, sizes and seeds (blocks: 4,096 super-blocks or 32,768 32-value blocks a format)
struct Format {
    const char* name;
    int B;
    std::vector<int> h16;
    int nblocks;
    unsigned seed;
};
inline std::vector<Format> formats() {
    constexpr int n = 4096;
    return {{"Q4_0", (int) sizeof(block_q4_0), {0}, n * 8, 1},          {"Q5_1", (int) sizeof(block_q5_1), {0, 2}, n * 8, 2},
            {"Q8_0", (int) sizeof(block_q8_0), {0}, n * 8, 3},          {"Q4_K", (int) sizeof(block_q4_K), {0, 2}, n, 4},
            {"Q5_K", (int) sizeof(block_q5_K), {0, 2}, n, 5},           {"Q6_K", (int) sizeof(block_q6_K), {(int) sizeof(block_q6_K) - 2}, n, 6},
            {"IQ4_NL", (int) sizeof(block_iq4_nl), {0}, n * 8, 7},      {"IQ4_XS", (int) sizeof(block_iq4_xs), {0}, n, 8},
            {"IQ3_S", (int) sizeof(block_iq3_s), {0}, n, 9},            {"IQ3_XXS", (int) sizeof(block_iq3_xxs), {0}, n, 10},
            {"IQ2_S", (int) sizeof(block_iq2_s), {0}, n, 11}};
}

// one group's line
struct Line {
    std::string family, format;
    int group = 0, dots = 0;
    uint64_t in_digest = 0, out_digest = 0;
    std::string reason;
};
inline std::string line_text(const Line& l) {
    char b[256];
    std::snprintf(b, sizeof b, "%s %s %d %d %016llx %016llx", l.family.c_str(), l.format.c_str(), l.group, l.dots,
                  (unsigned long long) l.in_digest, (unsigned long long) l.out_digest);
    return std::string(b) + (l.reason.empty() ? "" : " % " + l.reason);
}

// the dots of format T on the card (ours), as floats in dot order; empty on a CUDA error
template <GgmlType T> std::vector<float> run_ours(const Inputs& in, int nblocks) {
    const int ndots = nblocks * (qtraits<T>::qi / Dot<T>::vdr);
    char* dw = nullptr;
    block_q8_1* dy = nullptr;
    float* dout = nullptr;
    std::vector<float> out((size_t) ndots);
    bool ok = cudaMalloc(&dw, in.w.size()) == cudaSuccess && cudaMalloc(&dy, in.y.size() * sizeof(block_q8_1)) == cudaSuccess &&
              cudaMalloc(&dout, out.size() * 4) == cudaSuccess;
    if (ok) {
        cudaMemcpy(dw, in.w.data(), in.w.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, in.y.data(), in.y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice);
        k_ours<T><<<(ndots + 255) / 256, 256>>>(dw, dy, nblocks, dout);
        ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dw); cudaFree(dy); cudaFree(dout);
    if (!ok) out.clear();
    return out;
}

// the groups of one format from its dots: kGroupDots each, input digest = the whole format's inputs (weights then activations)
inline std::vector<Line> group_lines(const char* family, const char* format, const Inputs& in, const std::vector<float>& dots) {
    const uint64_t ind = fnv1a64(in.y.data(), in.y.size() * sizeof(block_q8_1), fnv1a64(in.w.data(), in.w.size()));
    std::vector<Line> v;
    for (size_t g0 = 0, gi = 0; g0 < dots.size(); g0 += kGroupDots, ++gi) {
        const size_t n = std::min<size_t>(kGroupDots, dots.size() - g0);
        Line l;
        l.family = family; l.format = format; l.group = (int) gi; l.dots = (int) n;
        l.in_digest = ind;
        l.out_digest = fnv1a64(dots.data() + g0, n * 4);
        v.push_back(l);
    }
    return v;
}

// every format's dots on this card, in the file's order (fn: run_ours<T> or an oracle-checked variant with the same signature)
template <class Fn> bool for_each_format(Fn&& fn) {
    bool ok = true;
    const std::vector<Format> fs = formats();
    auto one = [&](auto tag, const Format& f) {
        constexpr GgmlType T = decltype(tag)::value;
        const Inputs in = make_inputs(f.B, f.h16, f.nblocks, qtraits<T>::qk, f.seed);
        ok &= fn.template operator()<T>(f, in);
    };
    one(std::integral_constant<GgmlType, T_Q4_0>{}, fs[0]);
    one(std::integral_constant<GgmlType, T_Q5_1>{}, fs[1]);
    one(std::integral_constant<GgmlType, T_Q8_0>{}, fs[2]);
    one(std::integral_constant<GgmlType, T_Q4_K>{}, fs[3]);
    one(std::integral_constant<GgmlType, T_Q5_K>{}, fs[4]);
    one(std::integral_constant<GgmlType, T_Q6_K>{}, fs[5]);
    one(std::integral_constant<GgmlType, T_IQ4_NL>{}, fs[6]);
    one(std::integral_constant<GgmlType, T_IQ4_XS>{}, fs[7]);
    one(std::integral_constant<GgmlType, T_IQ3_S>{}, fs[8]);
    one(std::integral_constant<GgmlType, T_IQ3_XXS>{}, fs[9]);
    one(std::integral_constant<GgmlType, T_IQ2_S>{}, fs[10]);
    return ok;
}

inline const char* kHeader =
    "# src/split/tools/qdot_golden.txt - reference digests of the split engine's decode dots (src/split/src/qdot.cuh); checked by split_qdot_golden\n"
    "# (src/split/tools/qdot_golden.cu), written by split_qdot_test --write-golden (src/split/tools/qdot_test.cu), which writes only when every dot\n"
    "# of qdot.cuh is bit for bit ggml-cuda's vec_dot_*_q8_1 (vecdotq.cuh, third_party/strata-ggml, llama.cpp pin 3cf03257f219afbe7334045ff7c6a06ac68c627d)\n"
    "# on the same inputs.  Inputs: fixed seeds per format (qdot_golden.cuh make_inputs); a group = 4096 dots in the decode kernels' call order.\n"
    "# Line: <family> <format> <group> <dots> <input digest> <output digest> [% reason]; digests FNV-1a 64 (inputs: weight bytes then q8_1 blocks of\n"
    "# the format; outputs: the group's float bits).  Family cuda: every group (written with nvcc --use_fast_math, the split's flags).  Another\n"
    "# family (hip) has a line only where its bits differ from the cuda line, with the reason after '%'.\n";

// the file's lines (comments skipped); false if it cannot be read
inline bool read_lines(const char* path, std::vector<Line>& out) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[1024];
    while (std::fgets(buf, sizeof buf, f)) {
        if (buf[0] == '#' || buf[0] == '\n') continue;
        char fam[32], fmt[32];
        int g = 0, n = 0;
        unsigned long long a = 0, b = 0;
        if (std::sscanf(buf, "%31s %31s %d %d %llx %llx", fam, fmt, &g, &n, &a, &b) != 6) { std::fclose(f); return false; }
        Line l;
        l.family = fam; l.format = fmt; l.group = g; l.dots = n; l.in_digest = a; l.out_digest = b;
        const char* pct = std::strchr(buf, '%');
        if (pct) { l.reason = pct + 1; while (!l.reason.empty() && (l.reason.back() == '\n' || l.reason.back() == '\r')) l.reason.pop_back(); }
        out.push_back(l);
    }
    std::fclose(f);
    return true;
}

}  // namespace fnx_golden
