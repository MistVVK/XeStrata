// tests/cuda/prefill_decode_golden.cu - the fused prompt kernels' weight decode, bit for bit, against stored digests
// (prefill_decode_golden.txt), the same file on every vendor.
//
// What it checks.  The fused prompt-expert kernels (src/prefill/moe_fused_iq.cu: CUDA's native_kernel, HIP's native_w11_kernel)
// share one decode: load_unit<T> reads a 32-value sub-block's bytes into registers and convert<T> turns them into 32 int8 and the
// sub-block's two scales.  This test compiles that decode as it is (moe_fused_iq.cu included with STRATA_FUSED_DECODE_ONLY: the
// file up to grid_src, nothing else), runs it on the card over seeded blocks of every format the build decodes, and hashes the
// outputs: the 8 int8 words, then the bits of s0 and s1, per sub-block, in block order.  Where CUDA and HIP take different code
// for the same result (the i-quants' signs: signed8 vs the gfx11 / gfx12 table; IQ4's 16-entry table: byte_perm vs amdgcn_perm)
// the digests show it.  The fused kernels' OUTPUTS are not compared across vendors (their float order and SwiGLU differ); this
// is the shared part, which must be the same bits everywhere.
//
// Inputs.  One generator per format (splitmix64, seeded with the FNV-1a of the format's name), every draw a statement of its own
// (no two draws in one expression: C++ leaves their order to the compiler, and g++ and clang differ), the bytes taken from each
// 64-bit draw lowest first.  A block's scale fields (fp16) are then set to finite normal values from further draws.
//
// Oracle.  Every format ggml knows is also compared value by value with ggml's dequantizer (to_float): our value
// s * q (+ s1 for the formats with a minimum) within 1e-6 of ggml's, relative to the size of the terms.  --write refuses to write
// a file unless that holds; the trellis types (unknown to the pinned ggml) have no oracle here (their parity is in
// prefill_fused_iq_test).
//
//   prefill_decode_golden <file>            check: every format this build decodes must match the file (exit 1 if not, 77 without a card)
//   prefill_decode_golden --write <file>    write the file (only if the oracle holds)
//   prefill_decode_golden --print           print the lines
#include "ggml.h"   // the oracle's ggml_type / ggml_get_type_traits (moe_fused_iq.cu no longer includes ggml.h: it uses strata/quant/types.hpp)
#define STRATA_FUSED_DECODE_ONLY 1
#include "../../src/prefill/moe_fused_iq.cu"

#include <cinttypes>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::prefill::fused {
namespace {

constexpr int kBlocks = 2048;   // blocks per format
constexpr int kGroup = 256;     // blocks per digest line

// sub-blocks of 32 values per block (load_unit's ib): the 256-value super-blocks 8, Q2_0 2, the trellis types 4, the 32-value formats 1
__host__ __device__ constexpr int nsub(int t) {
    return t == T_IQ4_NL || t == T_Q5_0 || t == T_Q5_1 || t == T_Q8_0 ? 1 : t == T_Q2_0 || t == T_NVFP4 ? 2 : is_tq(t) ? 4 : 8;
}

template <int T> __global__ void decode_probe_kernel(const uint8_t* __restrict__ blocks, int nblk, uint32_t* __restrict__ out) {
#if defined(__HIPCC__)
    constexpr int SGN = (T == T_IQ2_XXS || T == T_IQ2_XS || T == T_IQ2_S || T == T_IQ3_XXS || T == T_IQ3_S) ? sign_entries(T) : 1;
    __shared__ uint4 ssign[SGN];   // (the kernels' sign table, filled as native_w11_kernel fills it)
    if constexpr (SGN > 1)
        for (int i = threadIdx.x; i < SGN; i += blockDim.x) ssign[i] = sign_entry(i, SGN == 128);
    __syncthreads();
    const uint4* sg = ssign;
#else
    const uint4* sg = nullptr;
#endif
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    if constexpr (T == T_NVFP4) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_fp4[k] << (8 * (k & 3));
    }
    const int i =blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * nsub(T)) return;
    const int b = i / nsub(T), ib = i % nsub(T);
    constexpr int NW = raw_words(T) > 5 ? raw_words(T) : 5;
    uint32_t raw[NW] = {};
    load_unit<T>(blocks + (size_t) b * block_bytes(T), ib, raw);
    uint32_t q[8];
    float s0, s1;
    convert<T>(raw, (const uint8_t*) grid_src<T>(), kv, q, s0, s1, sg);
    uint32_t* o = out + (size_t) i * 10;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = q[k];
    o[8] = __float_as_uint(s0);
    o[9] = __float_as_uint(s1);
}

void ck(cudaError_t e, const char* what, bool) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Fmt {
    int t;
    const char* name;
    bool oracle;   // ggml knows the type (to_float)
};
const Fmt kFormats[] = {
    {T_IQ2_XXS, "IQ2_XXS", true}, {T_IQ2_XS, "IQ2_XS", true}, {T_IQ2_S, "IQ2_S", true},   {T_IQ3_XXS, "IQ3_XXS", true},
    {T_IQ3_S, "IQ3_S", true},     {T_IQ4_XS, "IQ4_XS", true}, {T_IQ4_NL, "IQ4_NL", true}, {T_Q2_0, "Q2_0", true},
    {T_Q4_K, "Q4_K", true},       {T_Q5_K, "Q5_K", true},     {T_Q5_0, "Q5_0", true},     {T_Q5_1, "Q5_1", true},
    {T_Q8_0, "Q8_0", true},       {T_Q6_K, "Q6_K", true},     {T_TQ2_T, "TQ2_T", false},  {T_TQK6, "TQK6", false},
    {T_TQK7, "TQK7", false},      {T_NVFP4, "NVFP4", true}};

// the formats whose decode this build compiles: every one (the trellis decode on HIP too since the gfx12 port; the check keeps the list
// so that a build without one says so instead of failing)
bool built_here(int t) {
    (void) t;
    return true;
}

template <int T> void launch_probe(const uint8_t* blocks, int nblk, uint32_t* out) {
    const int n = nblk * nsub(T);
    decode_probe_kernel<T><<<(n + 255) / 256, 256>>>(blocks, nblk, out);
}
void run_probe(int t, const uint8_t* blocks, int nblk, uint32_t* out) {
    switch (t) {
        case T_IQ2_XXS: launch_probe<T_IQ2_XXS>(blocks, nblk, out); break;
        case T_IQ2_XS: launch_probe<T_IQ2_XS>(blocks, nblk, out); break;
        case T_IQ2_S: launch_probe<T_IQ2_S>(blocks, nblk, out); break;
        case T_IQ3_XXS: launch_probe<T_IQ3_XXS>(blocks, nblk, out); break;
        case T_IQ3_S: launch_probe<T_IQ3_S>(blocks, nblk, out); break;
        case T_IQ4_XS: launch_probe<T_IQ4_XS>(blocks, nblk, out); break;
        case T_IQ4_NL: launch_probe<T_IQ4_NL>(blocks, nblk, out); break;
        case T_Q2_0: launch_probe<T_Q2_0>(blocks, nblk, out); break;
        case T_Q4_K: launch_probe<T_Q4_K>(blocks, nblk, out); break;
        case T_Q5_K: launch_probe<T_Q5_K>(blocks, nblk, out); break;
        case T_Q5_0: launch_probe<T_Q5_0>(blocks, nblk, out); break;
        case T_Q5_1: launch_probe<T_Q5_1>(blocks, nblk, out); break;
        case T_Q8_0: launch_probe<T_Q8_0>(blocks, nblk, out); break;
        case T_Q6_K: launch_probe<T_Q6_K>(blocks, nblk, out); break;
        case T_TQ2_T: launch_probe<T_TQ2_T>(blocks, nblk, out); break;
        case T_TQK6: launch_probe<T_TQK6>(blocks, nblk, out); break;
        case T_TQK7: launch_probe<T_TQK7>(blocks, nblk, out); break;
        case T_NVFP4: launch_probe<T_NVFP4>(blocks, nblk, out); break;
        default: throw std::runtime_error("no probe for this type in this build");
    }
}

uint64_t fnv_bytes(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
constexpr uint64_t kFnv0 = 1469598103934665603ull;

struct SplitMix {
    uint64_t s;
    uint64_t next() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

// a finite normal fp16 from one draw: sign bit 20, exponent 9..18 (2^-6 .. 2^3), mantissa bits 8..17 of the draw
uint16_t half_from(uint64_t r) {
    const uint16_t sign = (uint16_t) (((r >> 20) & 1u) << 15);
    const uint16_t ex = (uint16_t) (9 + (r % 10));
    const uint16_t man = (uint16_t) ((r >> 8) & 0x3FFu);
    return (uint16_t) (sign | (ex << 10) | man);
}
// the fp16 scale fields of a block: their byte offsets
std::vector<int> half_fields(int t) {
    if (t == T_Q4_K || t == T_Q5_K || t == T_Q5_1) return {0, 2};   // d and dmin (Q5_1: d and m)
    if (t == T_Q6_K) return {208};                                   // d at the block's end
    if (t == T_NVFP4) return {};                                     // four UE4M3 scale bytes (every byte value is a scale, 0x7F the NaN that reads as 0), no fp16 field
    return {0};
}

std::vector<uint8_t> make_blocks(const Fmt& f) {
    const size_t bs = (size_t) block_bytes(f.t), n = bs * kBlocks;
    std::vector<uint8_t> m(n);
    SplitMix g{fnv_bytes(kFnv0, (const uint8_t*) f.name, std::strlen(f.name))};
    for (size_t i = 0; i < n; i += 8) {
        const uint64_t r = g.next();
        for (size_t k = 0; k < 8 && i + k < n; ++k) m[i + k] = (uint8_t) (r >> (8 * k));
    }
    const std::vector<int> fields = half_fields(f.t);
    for (int b = 0; b < kBlocks; ++b)
        for (int off : fields) {
            const uint64_t r = g.next();
            const uint16_t h = half_from(r);
            m[(size_t) b * bs + (size_t) off] = (uint8_t) (h & 0xFF);
            m[(size_t) b * bs + (size_t) off + 1] = (uint8_t) (h >> 8);
        }
    return m;
}

struct Line {
    std::string fmt;
    int group = 0, blocks = 0;
    uint64_t in = 0, out = 0;
};
std::string str(const Line& l) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s %d %d %016" PRIx64 " %016" PRIx64, l.fmt.c_str(), l.group, l.blocks, l.in, l.out);
    return buf;
}

// ggml's values against ours, every sub-block of every block; returns the number of values off by more than 1e-6
int64_t oracle(const Fmt& f, const std::vector<uint8_t>& m, const std::vector<uint32_t>& o) {
    const auto* tr = ggml_get_type_traits((ggml_type) f.t);
    const int per = (int) tr->blck_size, bs = block_bytes(f.t);
    std::vector<float> v((size_t) per);
    int64_t bad = 0;
    for (int b = 0; b < kBlocks; ++b) {
        tr->to_float(m.data() + (size_t) b * bs, v.data(), per);
        for (int ib = 0; ib < nsub(f.t); ++ib) {
            const uint32_t* w = &o[((size_t) b * nsub(f.t) + ib) * 10];
            float s0, s1;
            std::memcpy(&s0, &w[8], 4);
            std::memcpy(&s1, &w[9], 4);
            for (int j = 0; j < 32; ++j) {
                const float q = (float) (int8_t) (w[j >> 2] >> (8 * (j & 3)));
                float mine, size;
                if (has_min(f.t)) { mine = s0 * q + s1; size = std::fabs(s0 * q) + std::fabs(s1); }
                else { const float s = per16(f.t) && j >= 16 ? s1 : s0; mine = s * q; size = std::fabs(mine); }
                const float ref = v[(size_t) (32 * ib + j)];
                if (!(std::fabs(mine - ref) <= 1e-6f * size) && mine != ref) {
                    if (bad < 3)
                        std::printf("  %s block %d sub-block %d value %d: ours %.9g ggml %.9g\n", f.name, b, ib, j, mine, ref);
                    ++bad;
                }
            }
        }
    }
    return bad;
}

}  // namespace
}  // namespace strata::prefill::fused

int main(int argc, char** argv) {
    using namespace strata::prefill::fused;
    std::string file;
    bool write = false, print = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--write") write = true;
        else if (a == "--print") print = true;
        else file = a;
    }
    if (!print && file.empty()) { std::fprintf(stderr, "usage: prefill_decode_golden <file> | --write <file> | --print\n"); return 2; }
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no GPU: skipped\n"); return 77; }
        std::vector<Line> lines;
        std::vector<std::string> not_built;
        int64_t oracle_bad = 0, oracle_vals = 0;
        for (const Fmt& f : kFormats) {
            if (!built_here(f.t)) { not_built.push_back(f.name); continue; }
            const std::vector<uint8_t> m = make_blocks(f);
            const size_t nout = (size_t) kBlocks * nsub(f.t) * 10;
            uint8_t* dm = nullptr;
            uint32_t* dout = nullptr;
            ck(cudaMalloc((void**) &dm, m.size() + 16), "malloc", true);
            ck(cudaMalloc((void**) &dout, nout * 4), "malloc", true);
            ck(cudaMemcpy(dm, m.data(), m.size(), cudaMemcpyHostToDevice), "upload", true);
            ck(cudaMemset(dout, 0xFF, nout * 4), "memset", true);
            run_probe(f.t, dm, kBlocks, dout);
            ck(cudaGetLastError(), "launch", true);
            std::vector<uint32_t> o(nout);
            ck(cudaMemcpy(o.data(), dout, nout * 4, cudaMemcpyDeviceToHost), "download", true);
            ck(cudaFree(dm), "free", true);
            ck(cudaFree(dout), "free", true);
            const size_t bs = (size_t) block_bytes(f.t), words = (size_t) nsub(f.t) * 10;
            for (int g = 0; g < kBlocks / kGroup; ++g) {
                Line l;
                l.fmt = f.name;
                l.group = g;
                l.blocks = kGroup;
                l.in = fnv_bytes(kFnv0, m.data() + (size_t) g * kGroup * bs, (size_t) kGroup * bs);
                l.out = fnv_bytes(kFnv0, (const uint8_t*) (o.data() + (size_t) g * kGroup * words), (size_t) kGroup * words * 4);
                lines.push_back(l);
            }
            if (f.oracle) {
                const int64_t b = oracle(f, m, o);
                oracle_bad += b;
                oracle_vals += (int64_t) kBlocks * nsub(f.t) * 32;
                if (b) std::printf("%s: %lld values differ from ggml's dequantizer\n", f.name, (long long) b);
            }
        }
        std::printf("oracle (ggml to_float): %lld of %lld values differ\n", (long long) oracle_bad, (long long) oracle_vals);
        if (!not_built.empty()) {
            std::printf("not decoded by this build:");
            for (const std::string& s : not_built) std::printf(" %s", s.c_str());
            std::printf("\n");
        }
        if (print)
            for (const Line& l : lines) std::printf("%s\n", str(l).c_str());
        if (write) {
            if (oracle_bad) { std::fprintf(stderr, "not written: the oracle does not hold\n"); return 1; }
            std::ofstream out(file, std::ios::binary);
            out << "# strata fused prompt kernels: weight decode digests (tests/cuda/prefill_decode_golden.cu), v1\n"
                << "# format group blocks input-fnv64 output-fnv64 (output: per 32-value sub-block 8 int8 words + the bits of s0, s1)\n";
            for (const Line& l : lines) out << str(l) << "\n";
            std::printf("written %s: %zu lines\n", file.c_str(), lines.size());
            return out ? 0 : 1;
        }
        if (file.empty()) return oracle_bad ? 1 : 0;
        std::ifstream in(file);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", file.c_str()); return 1; }
        std::map<std::string, Line> want;
        std::string s;
        while (std::getline(in, s)) {
            if (s.empty() || s[0] == '#') continue;
            std::istringstream is(s);
            Line l;
            std::string a, b;
            is >> l.fmt >> l.group >> l.blocks >> a >> b;
            l.in = std::stoull(a, nullptr, 16);
            l.out = std::stoull(b, nullptr, 16);
            want[l.fmt + " " + std::to_string(l.group)] = l;
        }
        int bad = 0, same = 0;
        for (const Line& l : lines) {
            const auto it = want.find(l.fmt + " " + std::to_string(l.group));
            if (it == want.end()) { std::printf("MISSING in the file: %s\n", str(l).c_str()); ++bad; continue; }
            if (it->second.in != l.in) { std::printf("INPUTS DIFFER (the generator, not the decode): %s / file %016" PRIx64 "\n", str(l).c_str(), it->second.in); ++bad; continue; }
            if (it->second.out != l.out) { std::printf("DECODE DIFFERS: %s / file %016" PRIx64 "\n", str(l).c_str(), it->second.out); ++bad; continue; }
            ++same;
        }
        std::printf("decode digests: %d of %zu groups equal the file, %d differ%s\n", same, lines.size(), bad,
                    not_built.empty() ? "" : " (formats not decoded by this build are not checked)");
        return bad || oracle_bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_decode_golden: %s\n", e.what());
        return 1;
    }
}
